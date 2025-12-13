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

from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def collate_fn(data_list: list[dict]) -> dict:
    # Filter out None values (skipped samples)
    data_list = [data for data in data_list if data is not None]
    
    if len(data_list) == 0:
        raise ValueError("All samples in batch were filtered out (likely due to length exceeding max_prompt_length)")
    
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error'):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        from verl.utils.fs import copy_local_path_from_hdfs
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        # Filter out prompts that are too long
        if self.filter_prompts:
            def check_prompt_length(doc):
                try:
                    chat = doc[prompt_key]
                    if tokenizer.chat_template:
                        prompt_with_chat_template = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
                    else:
                        prompt_with_chat_template = chat[0]['content'] if isinstance(chat, list) and len(chat) > 0 else str(chat)
                    
                    # Tokenize to check length
                    tokenized = tokenizer(prompt_with_chat_template, add_special_tokens=False, return_tensors='pt')
                    sequence_length = tokenized['input_ids'].shape[-1]
                    return sequence_length <= self.max_prompt_length
                except Exception as e:
                    # If there's any error, skip this sample
                    print(f"[WARNING] Error checking prompt length, skipping sample: {e}")
                    return False
            
            original_len = len(self.dataframe)
            self.dataframe = self.dataframe[self.dataframe.apply(check_prompt_length, axis=1)]
            filtered_len = len(self.dataframe)
            if original_len != filtered_len:
                print(f"[INFO] Filtered out {original_len - filtered_len} samples with prompts longer than {self.max_prompt_length} tokens")

        print(f'filter dataset len: {len(self.dataframe)}')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        if self.tokenizer.chat_template:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        else:
            prompt_with_chat_template = chat[0]['content'] if isinstance(chat, list) and len(chat) > 0 else str(chat)
        # prompt_with_chat_template = chat

        try:
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                             tokenizer=self.tokenizer,
                                                                             max_length=self.max_prompt_length,
                                                                             pad_token_id=self.tokenizer.pad_token_id,
                                                                             left_pad=True,
                                                                             truncation=self.truncation)
        except NotImplementedError as e:
            # If prompt is too long and truncation='error', skip this sample
            # Return None and let the DataLoader filter it out
            print(f"[WARNING] Skipping sample {item}: {e}")
            return None

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict
