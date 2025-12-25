# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ToolBench模式的PPO训练主文件
"""

from verl import DataProto
import torch
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import numpy as np

# 导入ToolBench reward manager
from search_r1.llm_agent.toolbench_reward import ToolBenchRewardManager


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        # Get GPU count from CUDA_VISIBLE_DEVICES or config
        import os
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            num_gpus = len(cuda_visible.split(','))
        else:
            num_gpus = config.trainer.n_gpus_per_node
        
        ray.init(
            num_gpus=num_gpus,
            num_cpus=num_gpus,  # Match CPU to GPU count
            runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}},
            ignore_reinit_error=True
        )

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # Reward model setup (if needed)
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id
    
    # 创建ToolBench Reward Manager
    # 从config中读取reward权重（如果配置了）
    format_reward_weight = getattr(config.reward_model, 'format_reward_weight', 0.1)
    function_call_reward_weight = getattr(config.reward_model, 'function_call_reward_weight', 0.2)
    finish_reward_weight = getattr(config.reward_model, 'finish_reward_weight', 0.3)

    pass_reward_weight = getattr(config.reward_model, 'pass_reward_weight', 0.1)
    reward_server_url = getattr(config.reward_model, 'reward_server_url', "http://localhost:8000/evaluate_batch")

    reward_fn = ToolBenchRewardManager(
        tokenizer=tokenizer,
        format_reward_weight=format_reward_weight,
        function_call_reward_weight=function_call_reward_weight,
        finish_reward_weight=finish_reward_weight,
        num_examine=1,
        pass_reward_weight=pass_reward_weight,
        reward_server_url=reward_server_url
    )

    # Validation使用相同的reward函数
    val_reward_fn = ToolBenchRewardManager(
        tokenizer=tokenizer,
        format_reward_weight=format_reward_weight,
        function_call_reward_weight=function_call_reward_weight,
        finish_reward_weight=finish_reward_weight,
        num_examine=1,  # 验证时打印更多信息
        pass_reward_weight=pass_reward_weight,
        reward_server_url=reward_server_url
    )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
