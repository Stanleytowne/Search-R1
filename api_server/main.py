from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.requests import Request
import uvicorn
import time
import asyncio
from datetime import datetime
import json
import os, yaml
import requests
from typing import Union
from utils import standardize, change_name

from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from tenacity import retry, wait_random_exponential, stop_after_attempt

import textwrap
import aiofiles
from colorama import init, Fore, Style

# Initialize colorama for cross-platform color support
init(autoreset=True)

# Color logging functions
def debug_print(*args, **kwargs):
    """Print DEBUG messages in cyan"""
    print(f"{Fore.CYAN}[DEBUG]{Style.RESET_ALL}", *args, **kwargs)

def error_print(*args, **kwargs):
    """Print ERROR messages in red"""
    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL}", *args, **kwargs)

def warn_print(*args, **kwargs):
    """Print WARN messages in yellow"""
    print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL}", *args, **kwargs)

def info_print(*args, **kwargs):
    """Print INFO messages in green"""
    print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL}", *args, **kwargs)

config_file='config.yml'
CONFIG = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
print(CONFIG)
CACHE_FOLDER = CONFIG['cache_folder']

# OpenAI API - Use AsyncOpenAI for concurrent requests
from openai import AsyncOpenAI
if 'api_base' in CONFIG:
    OPENAI_API_BASE=CONFIG['api_base']
else:
    OPENAI_API_BASE="https://api.openai.com/v1"
OPENAI_API_KEY=CONFIG['api_key']

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class Info(BaseModel):
    category: str
    tool_name: str
    api_name: str
    tool_input: Union[str, dict]
    strip: str

def prepare_tool_name_and_url(info):
    category = info.category
    standard_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in standard_category or "," in standard_category:
        standard_category = standard_category.replace(" ", "_").replace(",", "_")
    standard_category = standard_category.replace("__", "_")
    
    tool_name = info.tool_name
    # First standardize the tool_name to get the proper format
    if not tool_name.endswith(f"_for_{standard_category}"):
        tool_name = standardize(info.tool_name)
        tool_name += f"_for_{standard_category}"
    else:
        tool_name = standardize(tool_name)
    
    # Extract tool_name without category suffix for splitting
    tmp_tool_name = tool_name.replace(f"_for_{standard_category}", "")
    
    # Process api_name: remove _for_{tool_name} suffix if present
    api_name_raw = info.api_name
    # Try to remove the _for_{tool_name} pattern (with or without category suffix)
    if f"_for_{tmp_tool_name}" in api_name_raw:
        api_name = api_name_raw.split(f"_for_{tmp_tool_name}")[0]
    else:
        # Try with the original tool_name from info
        original_tool = standardize(info.tool_name)
        if f"_for_{original_tool}" in api_name_raw:
            api_name = api_name_raw.split(f"_for_{original_tool}")[0]
        else:
            api_name = api_name_raw
    
    # Apply standardization and name changes
    api_name = change_name(standardize(api_name))
    
    code_string = f"""from my_tools.{standard_category}.{tmp_tool_name}.api import {api_name}"""
    return tool_name, standard_category, api_name, code_string


@app.post('/virtual')
# @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(1))
async def get_virtual_response(request: Request, info: Info):

    print('#'*30)
    debug_print(f"Received request: category={info.category}, tool_name={info.tool_name}, api_name={info.api_name}")
    
    tool_name, standard_category, api_name, code_string = prepare_tool_name_and_url(info)
    tool_input = info.tool_input
    tool_name_original = info.tool_name

    if api_name == "chat_with_user":
        response_dict = {"error": "", "response": "Chat with user."}
        return response_dict
    
    try:
        tool_input = json.loads(tool_input)
    except Exception as e:
        if tool_input == "":
            tool_input = {}
        elif isinstance(tool_input, dict):
            tool_input = tool_input
        else:
            error_print(f"Can not parse tool input into json: {tool_input}")
            print(type(tool_input))
            print(tool_input)
            response_dict = {"error": f"Tool input parse error...\n", "response": ""}
            return response_dict
    if not os.path.exists(CACHE_FOLDER):
        os.mkdir(CACHE_FOLDER)

    # load from cache (async file I/O)
    cache = {}
    # prerequisite: to read files correctly, "my_tools_cache" folder and "toolenv/tools/" folder should be available
    try:
        cache_file_path = os.path.join(CACHE_FOLDER, standard_category, tool_name, api_name+".json")
        if os.path.exists(cache_file_path):
            async with aiofiles.open(cache_file_path, 'r') as f:
                content = await f.read()
                tools_cache_record = json.loads(content)
                cache.update(tools_cache_record)
                if str(tool_input) in cache:
                    debug_print("using cached real response")
                    response_dict = cache[str(tool_input)]
                    return response_dict
    except Exception as e:
        error_print(f"Loading cache error: {e}")

    # parse api_doc
    tool_name_original = standardize(tool_name_original)
    # Note: api_name is already processed by prepare_tool_name_and_url (change_name(standardize(...)))
    # Do NOT standardize it again here
    api_doc = {
        'tool_description': "",
        'api_info': "",
    }
    try:
        tools_json_path = os.path.join(CONFIG['tools_folder'], standard_category, tool_name_original.split("_for_")[0]+".json")
        debug_print(f"tools_json_path: {tools_json_path}")
        if os.path.exists(tools_json_path):
            # read json (async file I/O)
            async with aiofiles.open(tools_json_path, 'r') as f:
                content = await f.read()
                api_intro = json.loads(content)
            # get tool_dexcription and api_info
            tool_description = api_intro['tool_description']
            api_info = []
            available_api_names = []
            for api in api_intro['api_list']:
                # Match using the same normalization as prepare_tool_name_and_url
                # which uses change_name(standardize(...))
                normalized_api_name = change_name(standardize(api['name']))
                available_api_names.append(f"{api['name']} -> {normalized_api_name}")
                if api_name == normalized_api_name:
                    api_info.append(api)
            # check invalid api name
            if len(api_info) == 0:
                error_print(f"cannot match api name: looking for '{api_name}'. Available APIs (original -> normalized): {available_api_names[:10]}...")
                # Return error response if API name cannot be matched
                return {"error": f"Cannot match API name: '{api_name}'. Available APIs (first 5): {[name.split(' -> ')[1] for name in available_api_names[:5]]}", "response": ""}
            api_doc = {
                'tool_description': tool_description,
                'api_info': api_info
            }
        else:
            error_print(f"cannot get {tool_name_original}")
            # Return error response if tool file doesn't exist
            return {"error": f"Cannot find tool definition file for: {tool_name_original}", "response": ""}
    except Exception as e:
        error_print(f"loading api_doc error: {e}")
        # Return error response if there's an exception loading api_doc
        return {"error": f"Error loading API documentation: {str(e)}", "response": ""}

    # get several examples from cache
    example_num = 5
    # get top example_num examples
    api_example = list(cache.items())[:example_num]
    while len(str(api_example)) > 2048 and example_num > 1:
        example_num -= 1
        api_example = list(cache.items())[:example_num]

    # Additional check: ensure api_info is not empty before calling the function
    if not api_doc.get('api_info') or len(api_doc['api_info']) == 0:
        return {"error": f"API information is empty for API: {api_name}", "response": ""}
    
    debug_print(f"api example: {api_example}")
    debug_print(f"tool_input: {tool_input}")
    debug_print(f"api_doc: {api_doc}")
        
    result = await fake_response_function_chat(api_example,tool_input,api_doc)

    if CONFIG['is_save']:
        await save_cache(cache, tool_input, result, standard_category, tool_name, api_name)
    if not isinstance(result, dict):
        return json.loads(result)
    else:
        return result
    
def is_valid_json(result):
    """
    Checks if the given string is valid JSON.

    Args:
      data: The string to be checked.

    Returns:
      True if the string is valid JSON, False otherwise.
    """
    # check json format
    try:
        result = json.loads(result)
        return True
    except Exception as e:
        error_print(f"Can not parse result into json: {result}")
        return False

def check_result(processes_value: dict):
    if 'error' not in processes_value or processes_value['error'] != '':
        return False
    if 'response' not in processes_value:
        return False
    response = str(processes_value['response'])
    if 'got an unexpected keyword argument' in response.lower():
        return True
    elif 'rate limit' in response.lower() or 'time out' in response.lower() or 'timed out' in response.lower() or 'does not exist' in response.lower() or '404' in response.lower() or '504' in response.lower() or '500' in response.lower() or 'internal error' in response.lower() or 'API doesn\'t exists' in response.lower() or "API doesn\'t exists" in response.lower() or response == '{\'message\': "API doesn\'t exists"}' or 'Service Not Found' in response:
        return False
    elif 'authoriz' in response.lower() or 'authenticat' in response.lower() or 'unauthorized' in response.lower() or 'blocked user' in response.lower() or 'unsubscribe' in response.lower() or 'blocked' in response.lower() or '401' in response.lower() or '403' in response.lower() or 'credential' in response.lower() or 'unauthenticated' in response.lower() or 'disabled for your subscription' in response.lower() or 'ACCESS_DENIED' in response or 'invalid consumer key' in response.lower():
        return False
    elif 'parameter' in response.lower() or 'parse' in response.lower() or 'is not defined' in response.lower():
        return False
    elif len(response) == 0:
        return False
    elif "status_code=50" in response or "status_code=429" in response:
        return False
    return True

async def save_cache(cache, tool_input, result, standard_category, tool_name, api_name, save_folder=CACHE_FOLDER):
    # save cache (async file I/O)
    try:
        if isinstance(result, dict):
            cache[str(tool_input)] = result
        elif isinstance(result, str):
            try:
                result_dict = json.loads(result)
                cache[str(tool_input)] = result_dict
            except Exception as e:
                error_print(f"Load result failed: {e}")
                return

        # Create directories if they don't exist
        category_dir = os.path.join(save_folder, standard_category)
        tool_dir = os.path.join(category_dir, tool_name)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir, exist_ok=True)
        if not os.path.exists(tool_dir):
            os.makedirs(tool_dir, exist_ok=True)
        
        # Write cache file asynchronously
        cache_file_path = os.path.join(tool_dir, api_name+".json")
        async with aiofiles.open(cache_file_path, 'w') as f:
            await f.write(json.dumps(cache, indent=4))
    except Exception as e:
        error_print(f"Save cache failed: {e}")

async def fake_response_function_chat(api_example, tool_input, api_doc):
    '''
    api_example: list of tuple, [(input, output), ...]
    tool_input: dict, input of the tool
    api_doc: dict, api document
    '''
    system_prompt = textwrap.dedent("""
        You are an advanced API Simulator and Validator. Your role is to act as a real API server, strictly adhering to the provided documentation to process requests.

        ### 1. Input Structure Explanation
        The user will provide input in the following specific format:
        API Documentation:
        Contains the API's URL, method, description, and parameter definitions (required/optional).
        API Examples:
        Contains reference calls (Note: This section is truncated to 2048 chars and may be incomplete; use it for style reference but rely on Documentation for logic).
        API Input:
        The specific arguments/payload you need to process.

        ### 2. Processing Logic
        1. **Analyze:** Read the `API Documentation` to understand the schema and constraints (types, required fields).
        2. **Validate:** Check the `API Input` against the `API Documentation`.
        - Are all `required_parameters` present?
        - Do the data types match (e.g., string vs int)?
        3. **Execute:**
        - **If Valid:** Generate a realistic, rich JSON response. 
        - **If Invalid:** Generate a JSON response where `error` describes the specific validation failure.

        ### 3. Output Format
        You must output ONLY a valid JSON object. No Markdown code blocks. No conversational text.

        **JSON Schema:**
        {
            "error": "String describing the error (if any), otherwise empty string",
            "response": <The_Simulated_Response_Object_or_Null>
        }

        ### 4. Behavior Rules
        - **Length Constraints:** - Keep the response **concise and lightweight**. Keep the entire JSON output shorter than 300 words. Do not generate excessively large payloads. If the API returns a list or array, **limit it to maximum 1-2 items**.
        - **Source of Truth:** Do not blindly copy "API Examples" if they contradict the "API Input". The "API Input" is your priority.
        """
    )
    system_prompt = {"role": "system", "content": system_prompt}

    # user prompt, truncated to 2048 characters if too long
    user_prompt = f"API Documentation: \n{str(api_doc)}\n\nAPI Examples: \n{str(api_example)[:2048]}\n\nAPI Input: \n{str(tool_input)}\n"
    user_prompt = {"role": "user", "content": user_prompt}

    # Use AsyncOpenAI for concurrent API calls
    client = AsyncOpenAI(
        api_key = OPENAI_API_KEY,
        base_url = OPENAI_API_BASE,
    )
    max_retries = 3 
    flag = False
    result = None
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model = CONFIG['model'],
                messages=[system_prompt, user_prompt],
                max_tokens = 1024,
                temperature=CONFIG['temperature'],
                response_format={"type": "json_object"},
            )
            result = response.choices[0].message.content
            
            # Print token usage information
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', 0)
                
                # Check for reasoning/thinking tokens (for reasoning models like o1, o3)
                reasoning_tokens = getattr(usage, 'completion_tokens_details', None)
                reasoning_tokens = getattr(reasoning_tokens, 'reasoning_tokens', None) if reasoning_tokens else None
                
                if reasoning_tokens is not None:
                    # For reasoning models: show reasoning tokens separately
                    info_print(f"OpenAI API token usage - prompt: {prompt_tokens}, reasoning: {reasoning_tokens}, completion: {completion_tokens}, total: {total_tokens}")
                else:
                    # For standard models: show standard token usage
                    info_print(f"OpenAI API token usage - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}")
            
            if "```json" in result:
                result = result.replace("```json", "").replace("```", "").strip()
            if is_valid_json(result):
                flag = True
                break
            warn_print(f"Invalid JSON response on attempt {attempt + 1}. Retrying...")
            await asyncio.sleep(1)  # Async sleep instead of time.sleep
        except Exception as e:
            error_print(f"OpenAI API call failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
            else:
                break

    debug_print(f"result: {result}")

    if flag:
        return result
    else:
        fake_error = {
            "error": "The API call failed. Please try again later.",
            "response": "",
        }
        return json.dumps(fake_error)

if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=CONFIG['port'])