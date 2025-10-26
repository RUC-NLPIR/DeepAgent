# run_web_thinker.py
import os
import json
import time
import re
from tqdm import tqdm
import numpy as np
import torch
import string
from typing import Optional, Tuple, List, Dict, Set
import argparse
import random
import asyncio
import aiohttp
import yaml
from transformers import AutoTokenizer
from openai import AsyncOpenAI
import sys
sys.path.append('./src')
from evaluate.evaluate_base import (
    run_evaluation, 
    extract_answer_fn,
    evaluate_predictions_toolhop
)
from prompts.prompts_deepagent import (
    BEGIN_TOOL_SEARCH,
    END_TOOL_SEARCH,
    BEGIN_TOOL_SEARCH_RESULT,
    END_TOOL_SEARCH_RESULT,
    BEGIN_TOOL_CALL,
    END_TOOL_CALL,
    BEGIN_TOOL_RESPONSE,
    END_TOOL_RESPONSE,
    FOLD_THOUGHT,
    get_helpful_tools_prompt,
    tool_response_analysis_prompt,
    get_tool_search_intent_instruction,
    get_tool_call_intent_instruction,
    get_folded_thought_instruction,
    get_episode_memory_instruction,
    get_working_memory_instruction,
    get_tool_memory_instruction,
    get_gpt_oss_system_prompt,
)
from prompts.prompts_deepagent import (
    main_reasoning_prompt_openset_general_qa,
    main_reasoning_prompt_closeset_general_qa,
    main_reasoning_prompt_closeset_embodied_task,
    main_reasoning_prompt_closeset_web_navigation,
    get_helpful_tools_prompt,
    tool_response_analysis_prompt,
    get_tool_search_intent_instruction,
    get_tool_call_intent_instruction,
    get_folded_thought_instruction,
    get_episode_memory_instruction,
    get_working_memory_instruction,
    get_tool_memory_instruction,
)
from prompts.task_specific_prompts import (
    get_toolhop_prompt,
)
from utils.utils import (
    extract_between,
    format_search_results,
)
from tools.tool_manager import ToolManager


def extract_json_from_response(response: str) -> str:
    """Extract JSON content from response using regex pattern matching."""
    try:
        # Pattern to match JSON content between ```json and ```
        pattern = r'```json\s*(.*?)\s*```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return response.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Search-o1 for various datasets and models.")
    parser.add_argument('--config_path', type=str, default='./config/base_config.yaml', help="Path to config YAML file.")
    parser.add_argument('--single_question', type=str, default=None, help="Single question to process instead of dataset")
    parser.add_argument('--dataset_name', type=str, required=False, default='custom', help="Name of the dataset to use.")
    parser.add_argument('--split', type=str, required=False, default='test', help="Dataset split to use.")
    parser.add_argument('--subset_num', type=int, default=-1, help="Number of examples to process. Defaults to all if not specified.")

    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument('--top_p', type=float, default=0.8, help="Top-p sampling parameter.")
    parser.add_argument('--top_k_sampling', type=int, default=20, help="Top-k sampling parameter.")
    parser.add_argument('--repetition_penalty', type=float, default=1.05, help="Repetition penalty. If not set, defaults based on the model.")
    parser.add_argument('--max_tokens', type=int, default=81920, help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset.")

    parser.add_argument('--enable_tool_search', action='store_true', default=False, help="Whether to enable tool search functionality.")
    parser.add_argument('--enable_thought_folding', action='store_true', default=False, help="Whether to enable thought folding functionality.")
    parser.add_argument('--max_action_limit', type=int, default=50, help="Maximum number of actions (tool search and tool call) per question.")
    parser.add_argument('--max_fold_limit', type=int, default=3, help="Maximum number of thought folds per question.")

    parser.add_argument('--top_k', type=int, default=10, help="Maximum number of search tools to return.")
    parser.add_argument('--use_jina', action='store_true', help="Whether to use Jina API for document fetching.")
    parser.add_argument('--jina_api_key', type=str, default='None', help="Your Jina API Key to Fetch URL Content.")
    parser.add_argument('--serper_api_key', type=str, default=None, help="Google Serper API key.")
    parser.add_argument('--eval', action='store_true', help="Whether to run evaluation after generation.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for generation. If not set, will use current timestamp as seed.")
    parser.add_argument('--concurrent_limit', type=int, default=32, help="Maximum number of concurrent API calls")
    return parser.parse_args()



async def generate_response(
    client: AsyncOpenAI,
    tokenizer: AutoTokenizer,
    prompt: str,
    semaphore: asyncio.Semaphore,
    generate_mode: str = "chat",
    think_mode: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 32768,
    repetition_penalty: float = 1.0,
    top_k: int = 1,
    model_name: str = "QwQ-32B",
    stop: List[str] = [],
    timeout: int = 3600,
    retry_limit: int = 3,
) -> Tuple[str, str]:
    """Generate a single response with retry logic"""
    for attempt in range(retry_limit):
        try:
            async with semaphore:
                if generate_mode == "chat":
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    # Check if we need to add <think> token for reasoning models
                    if think_mode and "<think>\n" not in formatted_prompt:
                        formatted_prompt = formatted_prompt + "<think>\n"
                    if not think_mode and "<think>\n" in formatted_prompt:
                        formatted_prompt = formatted_prompt.replace("<think>\n", "\n")
                else:
                    formatted_prompt = prompt

                response = await client.completions.create(
                    model=model_name,
                    prompt=formatted_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop,
                    extra_body={
                        'top_k': top_k,
                        'include_stop_str_in_output': True,
                        'repetition_penalty': repetition_penalty,
                    },
                    timeout=timeout,
                )
                return formatted_prompt, response.choices[0].text
        except Exception as e:
            print(f"Generate Response Error occurred: {e}, Starting retry attempt {attempt + 1}")
            # print(prompt)
            if "maximum context length" in str(e).lower():
                # If length exceeds limit, reduce max_tokens by half
                max_tokens = max_tokens // 2
                print(f"Reducing max_tokens to {max_tokens}")
            if attempt == retry_limit - 1:
                print(f"Failed after {retry_limit} attempts: {e}")
                return "", ""
            await asyncio.sleep(1 * (attempt + 1))
    return "", ""



async def run_tool_selection(
    client: AsyncOpenAI,
    tokenizer: AutoTokenizer,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    query: str,
    current_output: str,
    tool_search_result: List[Dict],
) -> dict:
    """
    Extract helpful tools.
    """
    previous_thoughts = current_output.split("\n\n")
    previous_thoughts = [f"Step {i+1}: {step}" for i, step in enumerate(previous_thoughts)]
    previous_thoughts = "\n\n".join(previous_thoughts[-10:])

    search_intent_prompt = get_tool_search_intent_instruction(previous_thoughts)
    _, search_intent = await generate_response(
        client=client,
        tokenizer=tokenizer,
        model_name=args.aux_model_name,
        prompt=search_intent_prompt,
        semaphore=semaphore,
        generate_mode="chat",
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k_sampling,
    )

    # Extract only the openai_function part for the prompt
    openai_functions_for_prompt = [tool['openai_function'] for tool in tool_search_result if 'openai_function' in tool]
    prompt = get_helpful_tools_prompt(
        query=query,
        search_intent=search_intent,
        tool_search_result=json.dumps(openai_functions_for_prompt, indent=2)
    )

    _, response = await generate_response(
        client=client,
        tokenizer=tokenizer,
        prompt=prompt,
        semaphore=semaphore,
        generate_mode="chat",
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k_sampling,
        model_name=args.aux_model_name,
    )

    match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        final_tools = match.group(1)
    else:
        # Fallback to returning the original tools if no final tools are found
        final_tools = json.dumps(openai_functions_for_prompt, indent=2)

    return final_tools.strip('\n ')


async def run_tool_response_analysis(
    client: AsyncOpenAI,
    tokenizer: AutoTokenizer,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    tool_call: dict,
    current_output: str,
    tool_response: str,
) -> str:
    """
    Analyze tool response and extract relevant information for the current task.
    """
    previous_thoughts = current_output.split("\n\n")
    previous_thoughts = [f"Step {i+1}: {step}" for i, step in enumerate(previous_thoughts)]
    previous_thoughts = "\n\n".join(previous_thoughts[-10:])

    tool_call_intent_prompt = get_tool_call_intent_instruction(previous_thoughts)
    _, tool_call_intent = await generate_response(
        client=client,
        tokenizer=tokenizer,
        model_name=args.aux_model_name,
        prompt=tool_call_intent_prompt,
        semaphore=semaphore,
        generate_mode="chat",
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k_sampling,
    )

    prompt = tool_response_analysis_prompt(
        tool_call=json.dumps(tool_call, indent=2),
        tool_call_intent=tool_call_intent,
        tool_response=tool_response
    )

    _, response = await generate_response(
        client=client,
        tokenizer=tokenizer,
        prompt=prompt,
        semaphore=semaphore,
        generate_mode="chat",
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k_sampling,
        model_name=args.aux_model_name,
    )
    
    return response


async def run_thought_folding(
    client: AsyncOpenAI,
    tokenizer: AutoTokenizer,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    question: str,
    current_output: str,
    interactions: List[Dict] = None,
    available_tools: List[Dict] = None,
) -> Tuple[str, str, str]:
    """
    Generate three types of memory in parallel: episode memory, working memory, and tool memory.
    """
    previous_thoughts = current_output.split("\n\n")
    previous_thoughts = [f"Step {i+1}: {step}" for i, step in enumerate(previous_thoughts)]
    previous_thoughts = "\n\n".join(previous_thoughts)

    # Prepare tool call history for tool memory
    tool_call_history = []
    if interactions:
        for interaction in interactions:
            if "tool_call_query" in interaction:
                tool_call_history.append({
                    "tool_call": interaction["tool_call_query"],
                    "tool_response": interaction["tool_response"]
                })

    # Prepare tool list for memory generation
    available_tools = ""
    if available_tools:
        # Convert available tools to JSON string for prompt
        available_tools = json.dumps(available_tools, indent=2)

    # Define async functions for each memory generation
    async def generate_episode_memory():
        episode_memory_prompt = get_episode_memory_instruction(question, previous_thoughts, available_tools)
        _, episode_memory_response = await generate_response(
            client=client,
            tokenizer=tokenizer,
            model_name=args.aux_model_name,
            prompt=episode_memory_prompt,
            semaphore=semaphore,
            generate_mode="chat",
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k_sampling,
        )
        return extract_json_from_response(episode_memory_response)

    async def generate_working_memory():
        working_memory_prompt = get_working_memory_instruction(question, previous_thoughts, available_tools)
        _, working_memory_response = await generate_response(
            client=client,
            tokenizer=tokenizer,
            model_name=args.aux_model_name,
            prompt=working_memory_prompt,
            semaphore=semaphore,
            generate_mode="chat",
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k_sampling,
        )
        return extract_json_from_response(working_memory_response)

    async def generate_tool_memory():
        tool_memory_prompt = get_tool_memory_instruction(question, previous_thoughts, json.dumps(tool_call_history, indent=2), available_tools)
        _, tool_memory_response = await generate_response(
            client=client,
            tokenizer=tokenizer,
            model_name=args.aux_model_name,
            prompt=tool_memory_prompt,
            semaphore=semaphore,
            generate_mode="chat",
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k_sampling,
        )
        return extract_json_from_response(tool_memory_response)

    # Generate all three memories in parallel
    episode_memory, working_memory, tool_memory = await asyncio.gather(
        generate_episode_memory(),
        generate_working_memory(),
        generate_tool_memory()
    )

    return episode_memory, working_memory, tool_memory


async def generate_main_reasoning_sequence(
    seq: Dict,
    client: AsyncOpenAI,
    aux_client: AsyncOpenAI,
    tokenizer: AutoTokenizer,
    aux_tokenizer: AutoTokenizer,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    tool_manager: ToolManager,
) -> Dict:
    """Process a single sequence through its entire reasoning chain with MAX_TOKENS limit"""
    
    # 初始化 token 计数器，初始值设为 prompt 的 token 数（简单用 split() 作为近似）
    MAX_TOKENS = 40000
    total_tokens = len(seq['prompt'].split())
    total_folds = 0
    seq['interactions'] = []
    
    # First response uses chat completion
    formatted_prompt, response = await generate_response(
        client=client,
        tokenizer=tokenizer, 
        think_mode=True,
        generate_mode="chat",
        model_name=args.model_name,
        prompt=seq['prompt'],
        semaphore=semaphore,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k_sampling,
        stop=[END_TOOL_SEARCH, END_TOOL_CALL, FOLD_THOUGHT],
    )
    
    # Update token count and sequence fields
    tokens_this_response = len(response.split())
    total_tokens += tokens_this_response
    
    seq['output'] += response.replace('</think>\n', '')
    seq['original_prompt'] = formatted_prompt
    seq['prompt'] = formatted_prompt + response.replace('</think>\n', '')
    
    while not seq['finished']:
        # Check if sequence is finished
        if not seq['output'].rstrip().endswith(END_TOOL_SEARCH) and not seq['output'].rstrip().endswith(END_TOOL_CALL) and not seq['output'].rstrip().endswith(FOLD_THOUGHT):
            seq['finished'] = True
            break
        
        tool_search_query = extract_between(response, BEGIN_TOOL_SEARCH, END_TOOL_SEARCH)
        tool_call_query = extract_between(response, BEGIN_TOOL_CALL, END_TOOL_CALL)
        if tool_search_query: tool_search_query = tool_search_query.replace("\n", "").strip()
        if tool_call_query: tool_call_query = tool_call_query.replace("\n", "").strip()
        
        seq['action_count'] += 1
        
        if seq['action_count'] < args.max_action_limit and total_tokens < MAX_TOKENS:

            if tool_search_query and len(tool_search_query) > 5 and seq['output'].rstrip().endswith(END_TOOL_SEARCH):
                if tool_search_query in seq['executed_search_queries']:
                    append_text = f"\n\n{BEGIN_TOOL_SEARCH_RESULT}You have already searched for this query.{END_TOOL_SEARCH_RESULT}\n\nHmm, I've already"
                    seq['prompt'] += append_text
                    seq['output'] += append_text
                    total_tokens += len(append_text.split())
                else:
                    if args.dataset_name == 'toolhop':
                        executable_tools = list(seq['item'].get('tools', {}).values())
                        initial_retrieved_tools = tool_manager.retrieve_tools(
                            tool_search_query,
                            args.top_k,
                            executable_tools
                        )
                    else:
                        initial_retrieved_tools = tool_manager.retrieve_tools(
                            tool_search_query,
                            args.top_k
                        )
                    seq['available_tools'].extend(initial_retrieved_tools)

                    helpful_tools = json.dumps([tool['openai_function'] for tool in initial_retrieved_tools if 'openai_function' in tool], indent=2)
                    if len(str(helpful_tools)) > 15000:
                        helpful_tools = await run_tool_selection(
                            client=aux_client,
                            tokenizer=aux_tokenizer,
                            semaphore=semaphore,
                            args=args,
                            query=tool_search_query,
                            current_output=seq['output'],
                            tool_search_result=initial_retrieved_tools,
                        )
                    
                    # Store web explorer input/output with all interactions
                    seq['interactions'].append({
                        "type": "tool_search",
                        "tool_search_query": tool_search_query,
                        # "initial_retrieved_tools": [json.dumps(tool['openai_function']) for tool in initial_retrieved_tools if 'openai_function' in tool],
                        "returned_tools": helpful_tools,
                    })
                    # Update sequence with search results
                    append_text = f"\n\n{BEGIN_TOOL_SEARCH_RESULT}{helpful_tools}{END_TOOL_SEARCH_RESULT}\n\n"
                    # if seq['action_count'] % 20 == 0 and total_folds < args.max_fold_limit:
                    #     append_text += "<system_message>You have made 20 actions. You can consider folding your thoughts and start a new round of reasoning.</system_message>\n\n"
                    seq['prompt'] += append_text
                    seq['output'] += append_text
                    seq['executed_search_queries'].add(tool_search_query)
                    total_tokens += len(append_text.split())

            elif tool_call_query and len(tool_call_query) > 5 and seq['output'].rstrip().endswith(END_TOOL_CALL):
                if tool_call_query in seq['executed_tool_calls'] and args.dataset_name not in ['alfworld', 'webshop']:
                    append_text = f"\n\n{BEGIN_TOOL_RESPONSE}You have already called this tool with the same arguments.{END_TOOL_RESPONSE}\n\nHmm, I've already"
                    seq['prompt'] += append_text
                    seq['output'] += append_text
                    total_tokens += len(append_text.split())
                else:
                    try:
                        tool_call_dict = json.loads(tool_call_query)
                        
                        # Adapt the model output to the format expected by the callers
                        adapted_tool_call = {
                            "function": {
                                "name": tool_call_dict.get("name"),
                                "arguments": tool_call_dict.get("arguments", {})
                            }
                        }
                        tool_response = await tool_manager.call_tool(adapted_tool_call, seq)
                        
                        if type(tool_response) in [dict, list]:
                            tool_response = json.dumps(tool_response)

                        # If the tool response is too long, analyze it and keep helpful information
                        if len(str(tool_response)) > 5000:
                            tool_response = await run_tool_response_analysis(
                                client=aux_client,
                                tokenizer=aux_tokenizer,
                                semaphore=semaphore,
                                args=args,
                                tool_call=adapted_tool_call,
                                current_output=seq['output'],
                                tool_response=tool_response,
                            )
                        
                        seq['interactions'].append({
                            "type": "tool_call",
                            "tool_call_query": tool_call_query,
                            "tool_response": tool_response
                        })
                        print(tool_call_query)
                        print(tool_response)
                        
                        append_text = f"\n\n{BEGIN_TOOL_RESPONSE}{tool_response}{END_TOOL_RESPONSE}\n\n"
                        # if seq['action_count'] >= 30 and total_folds < args.max_fold_limit:
                        #     append_text += "<system_message>You have made 30 actions. You can consider folding your thoughts and start a new round of reasoning.</system_message>\n\n"
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        total_tokens += len(append_text.split())

                        if seq['finished'] == True:
                            return seq

                    except Exception as e:
                        seq['interactions'].append({
                            "type": "tool_call",
                            "tool_call_query": tool_call_query,
                            "tool_response": {"error": f"Error calling tool: {e}"}
                        })
                        append_text = f"\n\n{BEGIN_TOOL_RESPONSE}Error calling tool: {e}{END_TOOL_RESPONSE}\n\n"
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        total_tokens += len(append_text.split())
                
                    seq['executed_tool_calls'].add(tool_call_query)
            
            elif seq['output'].rstrip().endswith(FOLD_THOUGHT):
                if total_folds >= args.max_fold_limit:
                    append_text = (
                        f"\n\n<system_message>You have reached the maximum number of allowed thought folds ({args.max_fold_limit}). "
                        "Further thought folding is not permitted. Please continue your reasoning based on your current information.</system_message>\n\n"
                        "Hmm, I've already"
                    )
                    seq['prompt'] += append_text
                    seq['output'] += append_text
                    total_tokens += len(append_text.split())
                else:
                    episode_memory, working_memory, tool_memory = await run_thought_folding(
                        client=aux_client,
                        tokenizer=aux_tokenizer,
                        semaphore=semaphore,
                        args=args,
                        question=seq['item']['Question'],
                        current_output=seq['output'],
                        interactions=seq['interactions'],
                        available_tools=seq['available_tools'],
                    )
                    append_text = f"Memory of previous folded thoughts:\n\nEpisode Memory:\n{episode_memory}\n\nWorking Memory:\n{working_memory}\n\nTool Memory:\n{tool_memory}"
                    seq['prompt'] = seq['original_prompt'].replace("Now, begin your reasoning for", f"{append_text}\n\nNow, begin your reasoning for")
                    seq['interactions'].append({
                        "type": "thought_folding",
                        "episode_memory": episode_memory,
                        "working_memory": working_memory,
                        "tool_memory": tool_memory,
                    })
                    print(seq['interactions'][-1])
                    total_tokens = len(seq['prompt'].split())
                    total_folds += 1
                    
            # Subsequent responses use completion mode
            _, response = await generate_response(
                client=client,
                tokenizer=tokenizer,
                model_name=args.model_name,
                prompt=seq['prompt'],
                semaphore=semaphore,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                repetition_penalty=args.repetition_penalty,
                top_k=args.top_k_sampling,
                stop=[END_TOOL_SEARCH, END_TOOL_CALL, FOLD_THOUGHT],
                generate_mode="completion"
            )
            
            # Update token count and sequence fields
            tokens_this_response = len(response.split())
            total_tokens += tokens_this_response
            
            seq['output'] += response.replace('</think>\n', '')
            seq['prompt'] += response.replace('</think>\n', '')

        else:
            if args.dataset_name in ['alfworld', 'webshop']:
                # For ALFWorld and WebShop, if actions are not allowed, the task completion is already determined, return directly
                return seq
            append_text = (
                f"\n\n<system_message>You have reached the maximum number of allowed actions ({args.max_action_limit}). "
                "You may no longer perform searches, call tools, or fold your thoughts. "
                "Please provide your final answer based on the information you have gathered so far."
                "</system_message>\n\nHmm, I've already"
            )
            seq['prompt'] += append_text
            seq['output'] += append_text
            
            _, final_response = await generate_response(
                client=client,
                tokenizer=tokenizer,
                model_name=args.model_name,
                prompt=seq['prompt'],
                semaphore=semaphore,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                repetition_penalty=args.repetition_penalty+0.1,
                top_k=args.top_k_sampling,
                generate_mode="completion",
            )
            
            seq['output'] += final_response
            seq['finished'] = True  # Sequence marked as finished
    
    return seq


async def main_async():
    print("Process started.")
    # ---------------------- Load args and config ----------------------
    args = parse_args()
    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    for k, v in config.items():  # Merge config into args (config values take precedence)
        setattr(args, k, v)

    # ---------------------- Initialize tokenizers ----------------------
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    aux_tokenizer = AutoTokenizer.from_pretrained(args.aux_tokenizer_path)

    # ---------------------- Set random seed ----------------------
    if args.seed is None:
        args.seed = int(time.time())
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ---------------------- Caching Mechanism ----------------------
    os.makedirs(args.tool_index_cache_dir, exist_ok=True)
    os.makedirs(args.search_cache_dir, exist_ok=True)

    # ---------------------- Initialize clients ----------------------
    semaphore = asyncio.Semaphore(args.concurrent_limit)
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )
    aux_client = AsyncOpenAI(
        api_key=args.aux_api_key,
        base_url=args.aux_base_url,
    )
    vqa_client = AsyncOpenAI(
        api_key=args.vqa_api_key,
        base_url=args.vqa_base_url,
    )

    # ---------------------- Initialize Tool Manager ----------------------
    tool_manager = await ToolManager.create(args)
    # Provide runtime clients to tool manager (for VQA and concurrency)
    tool_manager.set_runtime_clients(vqa_client=vqa_client, semaphore=semaphore, aux_client=aux_client, aux_model_name=args.aux_model_name)

    # ---------------------- Define output directory ----------------------
    if 'qwq' in args.model_name.lower():
        model_short_name = 'qwq'
        if '-v' in args.model_name.lower():
            model_short_name = args.model_name.lower()
    elif 'qwen3' in args.model_name.lower():
        if '32b' in args.model_name.lower():
            model_short_name = 'qwen3-32b'
        elif '8b' in args.model_name.lower():
            model_short_name = 'qwen3-8b'
        else:
            model_short_name = args.model_name.split('/')[-1].lower().replace('-instruct', '')
    elif 'deepseek' in args.model_name.lower():
        if 'qwen-7b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-7b'
        elif 'qwen-14b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-14b'
        elif 'qwen-32b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-32b'
        else:
            model_short_name = args.model_name.split('/')[-1].lower().replace('-instruct', '')
    else:
        model_short_name = args.model_name.split('/')[-1].lower().replace('-instruct', '')

    output_dir = f'./outputs/{args.dataset_name}.{model_short_name}.deepagent'
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------- Load and prepare data ----------------------
    if args.single_question:
        data_list = [{'Question': args.single_question}]
        args.dataset_name = 'custom'
    else:
        print('-----------------------')
        print(f'Using {args.dataset_name} dataset.')
        print('-----------------------')
        data_path = getattr(args, f"{args.dataset_name}_data_path", None)
        if not data_path or not os.path.exists(data_path):
            print(f"Data path for dataset '{args.dataset_name}' not found or configured.")
            return
        if args.dataset_name != 'api_bank':
            with open(data_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
            if args.dataset_name == 'alfworld':
                goal_to_subgoals = {item['goal']: item['subgoals'] for item in data_list}
            elif args.dataset_name == 'webshop':
                # For webshop, we create data based on the environment observations
                # Each session will have a different initial observation
                data_list = []
                for i in range(250):
                    data_list.append({
                        'id': i,
                        'session_id': f'fixed_{i}',
                    })
        if args.dataset_name == 'api_bank':
            # Load API-Bank data
            from tools.api_bank import APIBankDataLoader
            data_loader = APIBankDataLoader(args.api_bank_data_path)
            
            if not args.enable_tool_search:
                data_list = data_loader.load_level1_data()
            else:
                data_list = data_loader.load_level3_data()
        if args.subset_num != -1:
            if len(data_list) > args.subset_num:
                if args.dataset_name in ['alfworld', 'webshop']:
                    data_list = data_list[:args.subset_num]
                else:
                    data_list = random.sample(data_list, args.subset_num)

    # ---------------------- Prepare sequences ----------------------
    active_sequences = []
    for id, item in enumerate(data_list):
        question = ""
        tool_list = []
        tools_for_prompt = []

        if args.dataset_name == 'toolbench':
            from tools.rapid_api import api_json_to_openai_json, standardize
            question = item.get('query', '')
            if not args.enable_tool_search:
                # Convert ToolBench api_list format to OpenAI function format
                raw_api_list = item.get('api_list', [])
                tool_list = []
                for raw_api_json in raw_api_list:
                    # Add 'tool_name' if it's missing from a level, assuming it exists
                    if 'tool_name' not in raw_api_json and 'name' in raw_api_json:
                         raw_api_json['tool_name'] = raw_api_json['name'] # Fallback
                    
                    standard_tool_name = standardize(raw_api_json.get('tool_name', ''))
                    openai_function, _, _ = api_json_to_openai_json(raw_api_json, standard_tool_name)
                    tool_list.append(openai_function)
        
        elif args.dataset_name == 'toolhop':
            question = item.get('question', '')
            if not args.enable_tool_search:
                # For ToolHop, we need to add all_functions to each tool
                tools_dict = item.get('tools', {})
                functions_list = item.get('functions', [])
                tool_list = []
                for tool_spec in tools_dict.values():
                    tool_with_functions = tool_spec.copy()
                    tool_with_functions['all_functions'] = functions_list
                    tool_with_functions['tool_name'] = tool_spec.get('name', '')
                    tool_list.append(tool_with_functions)
                    tools_for_prompt.append(tool_spec)

        elif args.dataset_name == 'alfworld':
            from envs.alfworld import get_alfworld_function_definitions
            question = tool_manager.initial_obs_list[id]
            item['goal'] = question.split('Your task is to: ')[-1]
            item['subgoals'] = goal_to_subgoals[item['goal']]
            tool_list = get_alfworld_function_definitions()

        elif args.dataset_name == 'webshop':
            from envs.webshop import get_webshop_function_definitions
            question = tool_manager.initial_obs_list[id]
            item['Question'] = question
            tool_list = get_webshop_function_definitions()

        elif args.dataset_name in ['tmdb', 'spotify']:
            # RestBench datasets
            question = item.get('query', '')
            item['Question'] = question
            
            # Import and get RestBench tools
            from tools.restbench_api import get_restbench_tools
            tool_list = get_restbench_tools(args.dataset_name, args)
        elif args.dataset_name == 'api_bank':
            # API-Bank datasets
            if not args.enable_tool_search:
                # 构建从开始到最后一个User的完整对话作为question
                chat_history = item.get('chat_history', [])
                last_user_idx = -1
                for idx, turn in enumerate(chat_history):
                    if turn.get('role') == 'User':
                        last_user_idx = idx
                dialogue_turns = chat_history[: last_user_idx + 1] if last_user_idx != -1 else chat_history
                # 统一格式化为逐行的角色前缀文本
                question_lines = []
                for turn in dialogue_turns:
                    role = turn.get('role', 'User')
                    text = turn.get('text', '')
                    question_lines.append(f"{role}: {text}")
                question = "\n".join(question_lines).strip()

                # 从API调用中提取工具列表
                api_calls = item.get('api_calls', [])
                tool_list = []
                for api_call in api_calls:
                    api_name = api_call['api_name']
                    # 获取工具的OpenAI function格式
                    tool_info = tool_manager.caller.get_tool_info(api_name)
                    if tool_info:
                        tool_list.append(tool_info['openai_function'])
            else:  # api_bank_level3
                question = item.get('requirement', '')
                # Level-3需要工具搜索，不提供预定义工具列表
                tool_list = []
            
            item['Question'] = question

        elif args.dataset_name == 'gaia':
            # GAIA dataset
            from tools.tool_manager import get_gaia_tool_docs
            question = item.get('Question', item.get('question', item.get('query', '')))
            item['question'] = question
            if 'file_name' in item and item['file_name']:
                question += f"\n\nAttached file: {item['file_name']}"
                tool_list = get_gaia_tool_docs(task_type='file')
            elif ('problem_type' in item and item['problem_type'] == 'mm') or 'mm' in args.gaia_data_path:
                tool_list = get_gaia_tool_docs(task_type='mm')
            else:
                tool_list = get_gaia_tool_docs(task_type='text')
        
        elif args.dataset_name == 'hle':
            # HLE dataset
            from tools.tool_manager import get_hle_tool_docs
            question = item.get('Question', item.get('question', item.get('query', '')))
            if 'image' in item and item['image']:
                question += f"\n\nAttached image: {item['image']}"
                tool_list = get_hle_tool_docs(task_type='mm')
            else:
                tool_list = get_hle_tool_docs(task_type='text')

        elif args.dataset_name == 'browsecomp':
            # BrowseComp dataset: only provide web_search and browse_pages
            from tools.tool_manager import get_browsecomp_tool_docs
            question = item.get('Question', item.get('question', item.get('query', '')))
            tool_list = get_browsecomp_tool_docs()

        elif args.dataset_name == 'aime':
            # AIME dataset: only provide Python execution tool
            question = item.get('Question', item.get('question', item.get('query', '')))
            from tools.python_executor import get_openai_function_execute_python_code
            tool_list = [get_openai_function_execute_python_code(file_process=False)]

        else: # Generic case
            if 'Question' in item: question = item['Question']
            elif 'question' in item: question = item['question']
            elif 'query' in item: question = item['query']
        
        item['Question'] = question # Standardize question key

        if args.enable_tool_search:
            if args.dataset_name == 'toolhop':
                prompt = main_reasoning_prompt_openset_general_qa(question, get_toolhop_prompt())
            else:
                prompt = main_reasoning_prompt_openset_general_qa(question)
        else:
            if args.dataset_name == 'alfworld':
                prompt = main_reasoning_prompt_closeset_embodied_task(question, json.dumps(tool_list, indent=2))
            elif args.dataset_name == 'webshop':
                prompt = main_reasoning_prompt_closeset_web_navigation(question, json.dumps(tool_list, indent=2))
            elif args.dataset_name in ['tmdb', 'spotify']:
                # Prepend endpoint name + one-line description before tools JSON, similar to RestGPT style
                from tools.restbench_api import RestBenchAPITools, get_restbench_tools
                restbench_tools_helper = RestBenchAPITools(args.dataset_name, args)
                endpoint_lines = restbench_tools_helper.get_all_endpoints_summary()
                endpoints_block = "\n".join(endpoint_lines)
                tool_list = get_restbench_tools(args.dataset_name, args)
                tools_block = json.dumps(tool_list, indent=2) + "\n\nAvailable endpoints: " + endpoints_block
                prompt = main_reasoning_prompt_closeset_general_qa(question, tools_block)
            elif args.dataset_name == 'toolhop':
                prompt = main_reasoning_prompt_closeset_general_qa(question, json.dumps(tools_for_prompt, indent=2), get_toolhop_prompt())
            else:
                prompt = main_reasoning_prompt_closeset_general_qa(question, json.dumps(tool_list, indent=2))

        item['prompt'] = prompt

        seq_item = {
            'id': id,
            'item': item,
            'prompt': prompt,
            'output': '',
            'finished': False,
            'action_count': 0,
            'executed_search_queries': set(),
            'executed_tool_calls': set(),
            'success': False,
            'reward': 0.0,
        }
        if args.enable_tool_search:
            seq_item['available_tools'] = []
        else:
            seq_item['available_tools'] = tool_list
        
        active_sequences.append(seq_item)
    
    # ---------------------- Process all sequences ----------------------
    start_time = time.time()
    tasks = [
        generate_main_reasoning_sequence(
            seq=seq,
            client=client,
            aux_client=aux_client,
            tokenizer=tokenizer,
            aux_tokenizer=aux_tokenizer,
            semaphore=semaphore,
            args=args,
            tool_manager=tool_manager,
        )
        for seq in active_sequences
    ]
    
    # ---------------------- Run all sequences concurrently ----------------------
    with tqdm(total=len(tasks)) as pbar:
        async def track_progress(task):
            result = await task
            pbar.update(1)
            return result
        
        tracked_tasks = [track_progress(task) for task in tasks]
        completed_sequences = await asyncio.gather(*tracked_tasks)
    print(f"Total generation time: {time.time() - start_time} seconds")

    # ---------------------- Save results ----------------------
    t = time.localtime()
    if args.enable_tool_search:
        result_json_name = f'run.open.{t.tm_year}{t.tm_mon:02d}{t.tm_mday:02d}.{t.tm_hour:02d}{t.tm_min:02d}{t.tm_sec:02d}.json'
    else:
        result_json_name = f'run.close.{t.tm_year}{t.tm_mon:02d}{t.tm_mday:02d}.{t.tm_hour:02d}{t.tm_min:02d}{t.tm_sec:02d}.json'
    
    # Convert sets to lists before saving to avoid TypeError
    for seq in completed_sequences:
        if 'executed_search_queries' in seq:
            seq['executed_search_queries'] = list(seq['executed_search_queries'])
        if 'executed_tool_calls' in seq:
            seq['executed_tool_calls'] = list(seq['executed_tool_calls'])
    output_list = [item['output'] for item in completed_sequences]

    excluded_keys = ['Question']
    references = [{
        **{k: v for k, v in item['item'].items() if k not in excluded_keys},
        'output': item['output'],
        'action_count': item['action_count'],
        'executed_tool_searches': item['executed_search_queries'],
        'executed_tool_calls': item['executed_tool_calls'],
        'interactions': item.get('interactions', []),
        'success': item.get('success', False),
        'reward': item.get('reward', 0.0)  # reward for webshop
    } for item in completed_sequences]

    # ---------------------- Evaluate ----------------------
    if args.eval:
        output_metrics_path = result_json_name.replace('.json', '.metrics.json')
        output_metrics_overall_path = result_json_name.replace('.json', '.metrics.overall.json')
        if args.dataset_name == "toolhop":
            evaluate_predictions_toolhop(
                data=references,
                output_list=output_list,
                output_dir=output_dir,
                output_metrics_path=output_metrics_path,
                output_metrics_overall_path=output_metrics_overall_path,
            )
        elif args.dataset_name == "toolbench":
            from evaluate.evaluate_toolbench import compute_toolbench_metrics
            # Run ToolBench pass rate evaluation
            await compute_toolbench_metrics(
                data=references,
                client=aux_client,
                model_name=args.aux_model_name,
                max_eval_threads=args.concurrent_limit,
                evaluate_times=4,
                output_dir=output_dir,
                output_metrics_path=output_metrics_path,
                output_metrics_overall_path=output_metrics_overall_path,
            )
        elif args.dataset_name == 'alfworld':
            from evaluate.evaluate_alfworld import evaluate_predictions_alfworld
            evaluate_predictions_alfworld(
                data=references,
                output_list=output_list,
                output_dir=output_dir,
                output_metrics_path=output_metrics_path,
                output_metrics_overall_path=output_metrics_overall_path,
            )
        elif args.dataset_name == 'webshop':
            from evaluate.evaluate_webshop import evaluate_predictions_webshop
            evaluate_predictions_webshop(
                data=references,
                output_list=output_list,
                output_dir=output_dir,
                output_metrics_path=output_metrics_path,
                output_metrics_overall_path=output_metrics_overall_path,
            )
        elif args.dataset_name in ['tmdb', 'spotify']:
            from evaluate.evaluate_restbench import evaluate_restbench_predictions
            evaluate_restbench_predictions(
                data=references,
                output_list=output_list,
                output_dir=output_dir,
                output_metrics_path=output_metrics_path,
                output_metrics_overall_path=output_metrics_overall_path,
            )
        elif args.dataset_name == 'api_bank':
            from evaluate.evaluate_api_bank import evaluate_api_bank_predictions
            evaluate_api_bank_predictions(
                data=references,
                output_list=output_list,
                output_dir=output_dir,
                output_metrics_path=output_metrics_path,
                output_metrics_overall_path=output_metrics_overall_path,
                args=args
            )
        else:
            # For other datasets, use standard evaluation
            await run_evaluation(
                data=references,
                input_list=[item.get('question', item.get('Question', item.get('query', ''))) for item in references],
                output_list=output_list,
                output_dir=output_dir,
                output_metrics_path=output_metrics_path,
                output_metrics_overall_path=output_metrics_overall_path,
                use_llm=True,
                extract_answer=True,
                domain_fields=['Level', 'category', 'problem_type'],
                client=aux_client,
                semaphore=semaphore,
                model_name=args.aux_model_name,
            )
    
    # Save results if not evaluating
    if not args.eval:
        with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
            json.dump(references, json_file, indent=4, ensure_ascii=False)

    # ---------------------- Save caches ----------------------
    print("Saving web caches...")
    tool_manager.save_caches()

    print("Process completed.")

async def main():
    await main_async()

if __name__ == "__main__":
    asyncio.run(main_async())
