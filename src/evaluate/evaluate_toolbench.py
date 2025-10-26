# DeepAgent/src/evaluate/evaluate_toolbench.py

import json
import re
import random
import math
import os
import csv
import argparse
from typing import List, Union, Dict, Any, Optional
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import asyncio
from openai import AsyncOpenAI
import sys
sys.path.append('./src')
from prompts.prompts_tooleval import (
    CHECK_ANSWER_STATUS_PROMPT,
    PARSE_ANSWER_STATUS_PROMPT,
    CHECK_TASK_SOLVABLE_PROMPT,
    SELECT_BETTER_ANSWER_PROMPT
)
from evaluate.evaluate_base import extract_answer_fn

# Enums for evaluation status
class AnswerStatus:
    Unsure = "Unsure"
    Unsolved = "Unsolved"
    Solved = "Solved"
    
class TaskStatus:
    Unsure = "Unsure"
    Unsolvable = "Unsolvable"
    Solvable = "Solvable"
    
class AnswerPass:
    Unsure = "Unsure"
    Failed = "Failed"
    Passed = "Passed"


def change_name(name):
    """Change reserved names to avoid conflicts"""
    change_list = ["from", "class", "return", "false", "true", "id", "and", "", "ID"]
    if name in change_list:
        name = "is_" + name.lower()
    return name


def standardize(string):
    """Standardize string to valid identifier format"""
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+", "_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if len(string) > 0 and string[0].isdigit():
        string = "get_" + string
    return string


def process_name(name):
    """Process tool/API name using the same logic as Cal_path_rate.py"""
    return change_name(standardize(name))


def compute_path_rate(data: List[Dict[str, Any]]) -> float:
    """Compute path rate for ToolBench evaluation
    
    Path rate measures whether the model successfully used all relevant APIs for a given task.
    A path is considered successful if all relevant APIs from the ground truth were used.
    
    Args:
        data: List of model execution results, each containing 'execute_log' with 'api_result_ls'
        ground_truth_data: List of ground truth data, each containing 'relevant APIs'
    
    Returns:
        float: Path rate (number of successful paths / total paths)
    """
    if not data:
        return 0.0
    
    matching_count = 0
    total_count = len(data)
    
    for i in range(total_count):
        # Get API results from model execution
        api_calls = data[i].get("executed_tool_calls", [])
        relevant_apis = data[i].get("relevant APIs", [])
        
        # Extract API names from model execution results
        called_api_names = set()
        for api_call in api_calls:
            try:
                api_call = json.loads(api_call)
                tool_name = process_name(api_call.get('name', ''))
                called_api_names.add(tool_name)
            except:
                continue
        
        # Check ratio of relevant APIs used
        relevant_api_names = set([process_name(api_item[1]) + "_for_" + process_name(api_item[0]) for api_item in relevant_apis])
        ratio = len(relevant_api_names & called_api_names) / len(relevant_api_names)
        matching_count += ratio

        # # Check if all relevant APIs were used
        # all_apis_present = all(
        #     process_name(api_item[1]) + "_for_" + process_name(api_item[0]) in called_api_names 
        #     for api_item in relevant_apis
        # )
        # print("---\nrelevant_apis: ", set([process_name(api_item[1]) + "_for_" + process_name(api_item[0]) for api_item in relevant_apis]))
        # print("called_api_names: ", called_api_names)
        # print("all_apis_present: ", all_apis_present)
        
        # if all_apis_present:
        #     matching_count += 1
    
    path_rate = matching_count / total_count if total_count > 0 else 0.0
    return path_rate


class ToolBenchEvaluator:
    """ToolBench evaluator using DeepAgent's model"""
    
    def __init__(self, client: AsyncOpenAI, model_name: str):
        self.client = client
        self.model_name = model_name
        
    async def function_call(self, function_name: str, args: Dict, return_reason: bool = False) -> Dict:
        """Call evaluation functions using the model"""
        if function_name == 'check_answer_status':
            prompt = CHECK_ANSWER_STATUS_PROMPT.format(
                query=args['query'],
                answer=args['answer']
            )
        elif function_name == 'parse_answer_status':
            prompt = PARSE_ANSWER_STATUS_PROMPT.format(
                query=args['query'],
                answer=args['answer']
            )
        elif function_name == 'check_task_solvable':
            prompt = CHECK_TASK_SOLVABLE_PROMPT.format(
                task=args['task']
            )
        elif function_name == 'select_better_answer':
            prompt = SELECT_BETTER_ANSWER_PROMPT.format(
                query=args['query'],
                answer_0=args['answer_0'],
                answer_1=args['answer_1']
            )
        else:
            raise ValueError(f"Unknown function: {function_name}")
        
        # Generate response using model via chat.completions
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                # temperature=0.0,
                max_tokens=1024,
            )
            response_text = response.choices[0].message.content.strip()
            
            # Parse response based on function
            if function_name == 'check_answer_status':
                return self._parse_answer_status_response(response_text, return_reason)
            elif function_name == 'parse_answer_status':
                return self._parse_answer_status_response(response_text, return_reason)
            elif function_name == 'check_task_solvable':
                return self._parse_task_solvable_response(response_text, return_reason)
            elif function_name == 'select_better_answer':
                return self._parse_select_better_response(response_text)
        except Exception as e:
            print(f"Error in function call {function_name}: {e}")
            if function_name == 'select_better_answer':
                return {'index': '0'}  # Default fallback
            else:
                return self._get_default_response(function_name, return_reason)
    
    def _parse_answer_status_response(self, response: str, return_reason: bool) -> Dict:
        """Parse answer status response"""
        response_lower = response.lower()
        # Check 'unsolved' first to avoid 'solved' substring in 'unsolved'
        if 'unsolved' in response_lower:
            status = AnswerStatus.Unsolved
        elif 'solved' in response_lower:
            status = AnswerStatus.Solved
        else:
            status = AnswerStatus.Unsure
            
        result = {'answer_status': status}
        if return_reason:
            result['reason'] = response
        return result
    
    def _parse_task_solvable_response(self, response: str, return_reason: bool) -> Dict:
        """Parse task solvable response"""
        response_lower = response.lower()
        # Check 'unsolvable' first to avoid substring issues
        if 'unsolvable' in response_lower:
            status = TaskStatus.Unsolvable
        elif 'solvable' in response_lower:
            status = TaskStatus.Solvable
        else:
            status = TaskStatus.Unsure
            
        result = {'task_status': status}
        if return_reason:
            result['reason'] = response
        return result
    
    def _parse_select_better_response(self, response: str) -> Dict:
        """Parse select better answer response"""
        # Extract index from response
        index_match = re.search(r'index:\s*(\d+)', response.lower())
        if index_match:
            index = index_match.group(1)
        else:
            # Fallback: look for 0 or 1 in the response
            if '0' in response and '1' not in response:
                index = '0'
            elif '1' in response and '0' not in response:
                index = '1'
            else:
                index = '0'  # Default fallback
        return {'index': index}
    
    def _get_default_response(self, function_name: str, return_reason: bool) -> Dict:
        """Get default response when function call fails"""
        if function_name in ['check_answer_status', 'parse_answer_status']:
            result = {'answer_status': AnswerStatus.Unsure}
            if return_reason:
                result['reason'] = "Error in evaluation"
        elif function_name == 'check_task_solvable':
            result = {'task_status': TaskStatus.Unsure}
            if return_reason:
                result['reason'] = "Error in evaluation"
        else:
            result = {}
        return result
    
    def check_has_hallucination(self, available_tools: List[Dict], answer: Dict[Any, Any]) -> bool:
        """Check if the answer contains hallucinated tool usage"""
        available_names = set([tool['name'] for tool in available_tools])
        
        def check_node_valid(node: Dict) -> bool:
            if node['role'] == "tool":
                if isinstance(node['message'], dict):
                    node['message'] = str(node['message'])
                name_match = re.findall(r"'name':\s*'(.*?)'", node['message'], re.DOTALL)
                if name_match:
                    name = name_match[0]
                    return name in available_names
            return True
            
        def recursive_check(nodes: Union[List, Dict]) -> bool:
            if isinstance(nodes, Dict):
                if not check_node_valid(nodes):
                    return False
                else:
                    return recursive_check(nodes.get('next', []))
            if isinstance(nodes, List):
                for node in nodes:
                    if not recursive_check(node):
                        return False
                return True
            return True
            
        try:
            return recursive_check(answer.get('answer_details', []))
        except:
            return True
    
    async def check_is_solved(self, task_description: Dict, answer: Dict[Any, Any], return_reason: bool = False) -> Union[str, Optional[str]]:
        """Check if the task was solved"""
        # Derive final answer from output using extract_answer_fn
        raw_output = answer.get('output', '')
        final_answer_text = extract_answer_fn(raw_output, mode='qa', extract_answer=True) if isinstance(raw_output, str) else ''

        # Check for empty final answer
        if not final_answer_text:
            if return_reason:
                return AnswerStatus.Unsolved, "Empty final answer!"
            return AnswerStatus.Unsolved
        
        # First check
        ret = await self.function_call('check_answer_status', {
            'query': task_description['query'],
            'answer': final_answer_text
        }, return_reason)
        
        answer_status = ret['answer_status']  # keep as string
        
        if answer_status == AnswerStatus.Unsure:
            # Detailed check using both the extracted final answer and (optional) raw output for context
            enriched_answer_payload = json.dumps({
                'final_answer': final_answer_text,
                'output': raw_output
            })
            ret = await self.function_call('parse_answer_status', {
                'query': task_description['query'],
                'answer': enriched_answer_payload
            }, return_reason)
            answer_status = ret['answer_status']
        
        if return_reason:
            return answer_status, ret.get('reason', '')
        return answer_status
    
    async def check_task_solvable(self, task_description: Dict, has_been_solved: bool = False, return_reason: bool = False) -> Union[str, Optional[str]]:
        """Check if the task is solvable"""
        if has_been_solved:
            if return_reason:
                return TaskStatus.Solvable, 'Task has been solved before.'
            return TaskStatus.Solvable
        
        ret = await self.function_call('check_task_solvable', {
            'task': json.dumps(task_description)
        }, return_reason)
        
        task_status = ret['task_status']  # keep as string
        if return_reason:
            return task_status, ret.get('reason', '')
        return task_status
    
    def is_passed(self, task_description: Dict, answer: Dict[Any, Any], answer_status: str = None, task_status: str = None) -> str:
        """Determine if the answer passes evaluation"""
        if answer_status is None:
            return AnswerPass.Unsure
            
        if answer_status == AnswerStatus.Solved:
            return AnswerPass.Passed
        else:
            if task_status is None:
                return AnswerPass.Unsure
            
            if answer_status == AnswerStatus.Unsolved:
                if task_status == TaskStatus.Solvable:
                    return AnswerPass.Failed
                if task_status == TaskStatus.Unsure:
                    return AnswerPass.Unsure
                if task_status == TaskStatus.Unsolvable:
                    return AnswerPass.Passed
            elif answer_status == AnswerStatus.Unsure:
                if task_status == TaskStatus.Solvable:
                    return AnswerPass.Unsure
                if task_status == TaskStatus.Unsure:
                    return AnswerPass.Unsure
                if task_status == TaskStatus.Unsolvable:
                    return AnswerPass.Passed
                    
        return AnswerPass.Failed


async def compute_pass_rate(query_id: str, example: Dict, evaluator: ToolBenchEvaluator) -> tuple:
    """Compute pass rate for a single example"""
    try:
        # Check for hallucination
        not_hallucinate = evaluator.check_has_hallucination(
            example.get('api_list', []),
            example
        )
    except:
        not_hallucinate = True
    
    # Determine whether there is a final answer using extract_answer_fn on the raw output
    raw_output = example.get('output', '')
    extracted_final_answer = extract_answer_fn(raw_output if isinstance(raw_output, str) else '', mode='qa', extract_answer=True)
    
    # If no extracted final answer, treat as unsolved
    if not extracted_final_answer:
        return query_id, TaskStatus.Solvable, AnswerStatus.Unsolved, "failed", "No answer", not_hallucinate
    
    # Check if solved
    is_solved, is_solved_reason = await evaluator.check_is_solved(
        {
            'query': example['query'],
            'available_tools': example.get('api_list', []),
        },
        example,
        return_reason=True
    )
    
    is_solved_flag = (is_solved == AnswerStatus.Solved)
    
    # Check if task is solvable
    task_solvable, task_solvable_reason = await evaluator.check_task_solvable(
        {
            'query': example['query'],
            'available_tools': example.get('api_list', []),
        },
        has_been_solved=is_solved_flag,
        return_reason=True
    )
    
    # Determine if passed
    is_passed = evaluator.is_passed(
        {
            'query': example['query'],
            'available_tools': example.get('api_list', []),
        },
        example,
        answer_status=is_solved,
        task_status=task_solvable
    )
    
    reason = f"Is solved: {is_solved_reason}\nTask solvable: {task_solvable_reason}"
    
    if is_passed == AnswerPass.Passed:
        label = "passed"
    elif is_passed == AnswerPass.Failed:
        label = "failed"
    else:
        # If unsure, random choose
        label = "passed" if random.random() < 0.5 else "failed"
    
    return query_id, task_solvable, is_solved, label, reason, not_hallucinate, extracted_final_answer


async def compute_toolbench_metrics(
    data: List[Dict[str, Any]],
    client: AsyncOpenAI,
    model_name: str,
    max_eval_threads: int = 30,
    evaluate_times: int = 4,
    output_dir: str = None,
    output_metrics_path: str = None,
    output_metrics_overall_path: str = None,
) -> Dict[str, Any]:
    """Evaluate ToolBench pass rate for in-memory data and save metrics.

    Each item in `data` should contain at least:
      - 'query': str
      - 'available_tools': List[Dict]
      - 'answer_details': List or Dict
      - 'final_answer': str

    The function attaches an aggregated `metrics` key onto each item containing
    pass_rate and related fields, then saves the enriched dataset and overall
    metrics if output paths are provided.
    """

    # Compute path rate if ground truth data is provided
    path_rate = compute_path_rate(data)
    print(f"Computed path rate: {path_rate:.4f}")

    evaluator = ToolBenchEvaluator(client, model_name)

    semaphore = asyncio.Semaphore(max_eval_threads)

    async def eval_once(item_id: str, item: Dict[str, Any]):
        async with semaphore:
            return await compute_pass_rate(item_id, item, evaluator)

    tasks = []
    id_to_item: Dict[str, Dict[str, Any]] = {}

    for idx, item in enumerate(data):
        item_id = str(item.get('id', idx))
        id_to_item[item_id] = item
        for _ in range(evaluate_times):
            tasks.append(asyncio.create_task(eval_once(item_id, item)))

    label_cnt: Dict[str, Dict[str, int]] = {}
    last_status: Dict[str, Dict[str, Any]] = {}

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), ncols=100):
        try:
            query_id, task_solvable, is_solved, machine_label, reason, not_hallucinate, extracted_final_answer = await coro
        except Exception as e:
            # Skip erroneous task result and continue
            continue

        if query_id not in label_cnt:
            label_cnt[query_id] = {"passed": 0, "failed": 0}

        if machine_label == "passed":
            label_cnt[query_id]["passed"] += 1
        else:
            label_cnt[query_id]["failed"] += 1

        last_status[query_id] = {
            "task_solvable": task_solvable,
            "is_solved": is_solved,
            "reason": reason,
            "not_hallucinate": not_hallucinate,
            "extracted_final_answer": extracted_final_answer,
        }

    total_instances = len(data)
    passed_majority_count = 0
    pass_rates = []

    for idx, item in enumerate(data):
        item_id = str(item.get('id', idx))
        counts = label_cnt.get(item_id, {"passed": 0, "failed": 0})
        passed = counts["passed"]
        failed = counts["failed"]
        total = max(1, passed + failed)
        pass_rate = passed / total

        status = last_status.get(item_id, {})

        if passed > failed:
            majority_label = "passed"
            passed_majority_count += 1
        elif passed < failed:
            majority_label = "failed"
        else:
            # tie-breaker
            majority_label = "passed" if random.random() < 0.5 else "failed"
            if majority_label == "passed":
                passed_majority_count += 1

        pass_rates.append(pass_rate)

        item["metrics"] = {
            "pred_answer": status.get("extracted_final_answer", ""),
            "passed": passed,
            "failed": failed,
            "pass_ratio": pass_rate,
            "pass_rate_majority_voting": majority_label,
            "is_solved": status.get("is_solved", AnswerStatus.Unsure),
            "task_solvable": status.get("task_solvable", TaskStatus.Unsure),
            "reason": status.get("reason", ""),
            "not_hallucinate": bool(status.get("not_hallucinate", True)),
        }

    overall_metrics = {
        "total_instances": total_instances,
        # "average_pass_ratio": (sum(pass_rates) / len(pass_rates)) if pass_rates else 0.0,
        "average_success_rate": (passed_majority_count / total_instances) if total_instances else 0.0,
        "average_path_rate": path_rate,
    }
    print("ToolBench Evaluation Metrics:")
    print(overall_metrics)
    
    if output_dir:
        output_metrics_path = os.path.join(output_dir, output_metrics_path)
        output_metrics_overall_path = os.path.join(output_dir, output_metrics_overall_path)
    else:
        output_metrics_path = output_metrics_path
        output_metrics_overall_path = output_metrics_overall_path

    with open(output_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    with open(output_metrics_overall_path, 'w', encoding='utf-8') as f:
        json.dump(overall_metrics, f, ensure_ascii=False, indent=4)

    return overall_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ToolBench pass rate from an output JSON file")
    parser.add_argument('--input_json', type=str, required=True, help='Path to the input JSON file (list or dict) containing instances to evaluate')
    parser.add_argument('--model_name', type=str, default='qwen2.5-32b-instruct', help='Model name for evaluation')
    parser.add_argument('--base_url', type=str, default='http://10.148.8.235:8080/v1', help='Base URL for model API')
    parser.add_argument('--api_key', type=str, default='empty', help='API key for model')
    parser.add_argument('--max_eval_threads', type=int, default=50, help='Maximum number of concurrent evaluation tasks')
    parser.add_argument('--evaluate_times', type=int, default=4, help='Number of times to evaluate each example for majority voting')
    return parser.parse_args()


async def main():
    args = parse_args()
    print("Initializing API client...")
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    # Load input data
    if not os.path.exists(args.input_json):
        raise FileNotFoundError(f"Input JSON file not found: {args.input_json}")

    with open(args.input_json, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # Normalize to list of dicts and ensure each item has an 'id'
    data: List[Dict[str, Any]] = []
    if isinstance(raw, dict):
        for key, val in raw.items():
            if isinstance(val, dict):
                item = deepcopy(val)
                item.setdefault('id', str(key))
                data.append(item)
    elif isinstance(raw, list):
        for idx, val in enumerate(raw):
            if not isinstance(val, dict):
                continue
            item = deepcopy(val)
            item.setdefault('id', str(idx))
            data.append(item)
    else:
        raise ValueError("Unsupported input JSON format. Must be a list of dicts or a dict of id -> dict.")

    print(f"Loaded {len(data)} instances for evaluation.")

    # Run evaluation and save outputs
    overall_metrics = await compute_toolbench_metrics(
        data=data,
        client=client,
        model_name=args.model_name,
        max_eval_threads=args.max_eval_threads,
        evaluate_times=args.evaluate_times,
        output_dir=None,
        output_metrics_path=args.input_json.replace('.json', '.metrics.json'),
        output_metrics_overall_path=args.input_json.replace('.json', '.metrics.overall.json'),
    )

    print("Overall metrics:")
    print(overall_metrics)


if __name__ == "__main__":
    asyncio.run(main()) 