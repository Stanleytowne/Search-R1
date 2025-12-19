import requests
import json

url = 'http://10.153.48.58:8080/virtual'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

# Test cases extracted from toolllama_G123_dfs_eval.json with successful API responses
test_cases = [
    {
        "name": "Test 1: Tokopedia - Get sorting methods (successful response)",
        "data": {
            "category": "Data",
            "tool_name": "fresh_linkedin_profile_data",
            "api_name": "Get Open Profile Status",
            "tool_input": json.dumps({"linkedin_url": "https://www.linkedin.com/in/williamhgates/"}),
            "strip": "",
        }
    },
]

def run_test(test_case):
    """Run a single test case"""
    print(f"\n{'='*80}")
    print(f"{test_case['name']}")
    print(f"{'='*80}")
    print(f"Request data: {json.dumps(test_case['data'], indent=2)}")
    print(f"\nSending request...")
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(test_case['data']), timeout=60)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting API Server Tests")
    print(f"Server URL: {url}")
    
    results = []
    for test_case in test_cases:
        success = run_test(test_case)
        results.append((test_case['name'], success))
    
    # Print summary
    print(f"\n{'='*80}")
    print("Test Summary")
    print(f"{'='*80}")
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
