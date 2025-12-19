import requests
import json
import time

# 服务器配置
SERVER_URL = "http://localhost:1234/evaluate_batch"

def test_reward_server():
    # 模拟不同的测试用例
    test_cases = [
        {
            "name": "Solved Case (完全解决)",
            "query": "My company needs a new email address. Can you generate a new email address for us? Additionally, please provide the details of the latest email received on this address.",
            "trajectory": """Thought: We are currently calling the 'help_page_for_email_api' function with no arguments. The purpose of this function could be to provide help and information about the Email API. By calling this function, we can access the necessary documentation and guidance to assist in generating a new email address and retrieving the details of the latest email received.
Action: help_page_for_email_api
Action Input: {}

Observation: {"error": "Invalid API name: 'help_page_for_email_api'. Please check the API name.", "response": ""}

Hmm, it seems that there was an issue with the API called 'help_page_for_email_api'. The response is empty, indicating that the request failed. To proceed, let's try using a different API for generating a new email address and retrieving the details of the latest email received. By doing so, we can ensure that we get the information we need to fulfill the user's request.
Action: getnewemail10_for_apimail10
Action Input: {}

My previous action is invalid. I should organize my output into three parts: Thought, Action, and Action Input, and in the Action part, I should directly write the name of the API.

Thought: Based on the user's request, they want to generate a new email address for their company and also retrieve details of the latest email received on this address. To proceed, I will try calling the 'getnewemail10_for_apimail10' function, which is specifically designed to generate a new email address. This way, I can provide the user with a brand new email address for their company. By using this function, I can fulfill the user's request accurately and effectively.
Action: getnewemail10_for_apimail10
Action Input: {}

Observation: {"error": "", "response": {"email": "example123@tempmail.com"}}

Thought: No API-related errors were encountered after calling the 'getnewemail10_for_apimail10' function with no arguments. It seems like the API call was successful, and I can proceed with the task by using this new email address. We can now return the details of the latest email received on this address using the 'getnewemail10_for_apimail10' function again.
Action: getnewemail10_for_apimail10
Action Input: {}

Observation: {"error": "", "response": {"email": "example123@tempmail.com"}}

Thought: Based on the user's request, they want to generate a new email address for their company. To fulfill this, I called the 'getnewemail10_for_apimail10' function, which successfully generated a new email address and provided it to the user. The user can now use this email address for their company. Additionally, the API call failed for the 'help_page_for_email_api' function, which indicated an invalid API name. To avoid this issue, I will try using a different API for future assistance.
Action: help_page_for_email_api
Action Input: {}

Observation: {"error": "Invalid API name: 'help_page_for_email_api'. Please check the API name.", "response": ""}

Thought: I see that the API called 'help_page_for_email_api' failed with an "Invalid API name" error. To handle this situation, I will try using a different API for generating a new email address and retrieving the details of the latest email received. By doing so, I can ensure that the user's request is fulfilled effectively.
Action: getnewemail_for_apimail10
Action Input: {}"""
        },
    ]

    # 提取 queries 和 trajectories
    queries = [tc["query"] for tc in test_cases]
    trajectories = [tc["trajectory"] for tc in test_cases]

    payload = {
        "queries": queries,
        "trajectories": trajectories
    }

    print(f"🚀 发送 Batch 请求 (Size: {len(test_cases)}) 到 {SERVER_URL}...")
    
    start_time = time.time()
    try:
        response = requests.post(SERVER_URL, json=payload, timeout=60)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            results = response.json()
            scores = results.get("scores", [])
            
            print(f"✅ 请求成功！总耗时: {duration:.2f}s (平均每条: {duration/len(test_cases):.2f}s)")
            print("-" * 50)
            
            for i, tc in enumerate(test_cases):
                print(f"用例: {tc['name']}")
                print(f"得分: {scores[i]}")
                # 注意：如果在 server 端 log 了 reason，可以尝试让 server 也返回 reason 字段以便调试
                print("-" * 30)
        else:
            print(f"❌ 请求失败，状态码: {response.status_code}")
            print(f"错误详情: {response.text}")

    except Exception as e:
        print(f"🚨 发生异常: {e}")

if __name__ == "__main__":
    test_reward_server()