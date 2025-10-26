import json
import os
import logging
import re
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np
import sys
sys.path.append("./src/")
from tools.api_bank import (
    APIBankDataLoader, 
    APIBankExecutor, 
    APIBankRetriever,
    parse_api_call, 
    get_api_call, 
    calculate_rouge_l_score
)


def _check_api_correctness_with_tool(apis_dir: str, tool_name: str, pred_result: Dict, gt_result: Dict) -> bool:
    """Use each tool's built-in check_api_call_correctness to determine correctness, maintaining consistency with standard evaluation."""
    try:
        import importlib.util
        for file in os.listdir(apis_dir):
            if file.endswith('.py') and file not in ['__init__.py', 'api.py', 'tool_search.py']:
                api_file = file.split('.')[0]
                module_path = os.path.join(apis_dir, file)
                spec = importlib.util.spec_from_file_location(api_file, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                tool_class = getattr(module, tool_name, None)
                if tool_class and hasattr(tool_class, 'check_api_call_correctness'):
                    tool_instance = tool_class()
                    return bool(tool_instance.check_api_call_correctness(pred_result, gt_result))
    except KeyError:
        # Consistent with standard implementation: tool validation throwing KeyError is considered an error
        return False
    except Exception:
        # Other exceptions are uniformly treated as errors, classified by upper layer statistics
        return False
    return False


def evaluate_api_bank_level1(
    data: List[Dict],
    output_list: List[str],
    output_dir: str,
    output_metrics_path: str,
    output_metrics_overall_path: str,
    apis_dir: str
):
    """Evaluate API-Bank Level-1 (scenario with given candidate APIs)"""
    
    print("Evaluating API-Bank Level-1...")
    
    # Initialize tool executor (no longer used for execution, only for tool information)
    executor = APIBankExecutor(apis_dir=apis_dir)
    
    total_api_calls = 0
    correct_api_calls = 0
    total_rouge_score = 0
    detailed_items: List[Dict[str, Any]] = []
    
    error_statistics = {
        'NO_API_CALL': {'count': 0, 'samples': []},
        'API_NAME_MISMATCH': {'count': 0, 'samples': []},
        'HAS_EXCEPTION': {'count': 0, 'samples': []},
        'INPUT_MISMATCH': {'count': 0, 'samples': []},
        'OUTPUT_MISMATCH': {'count': 0, 'samples': []},
        'INVALID_INPUT_PARAMETER': {'count': 0, 'samples': []},
        'KEY_ERROR': {'count': 0, 'samples': []},
        'FAILED_PARSE_API_CALL': {'count': 0, 'samples': []},
        'MISS_INPUT_ARGUMENT': {'count': 0, 'samples': []},
    }
    
    for i, (item, output) in tqdm(enumerate(zip(data, output_list))):
        file_name = item.get('file', f'sample_{i}')
        
        # Parse conversation history
        chat_history = item.get('chat_history', [])
        
        # Calculate Rouge-L score for final response: use the last AI reply in conversation as reference, final boxed answer in model output as prediction
        gt_response = ""
        for turn in reversed(chat_history):
            if turn.get('role') == 'AI' and turn.get('text'):
                gt_response = turn['text']
                break
        # Extract boxed final answer from model output (supports both \\boxed{...} and \\boxed (...) formats)
        boxed_match = None
        try:
            boxed_match = re.search(r"\\boxed\s*\{([^}]*)\}", output)
            if not boxed_match:
                boxed_match = re.search(r"\\boxed\s*\(([^)]*)\)", output)
        except Exception:
            boxed_match = None
        final_answer = boxed_match.group(1).strip() if boxed_match else output
        
        rouge_score = 0.0
        if gt_response:
            rouge_score = calculate_rouge_l_score(gt_response, final_answer)
            total_rouge_score += rouge_score
        
        # Extract executed API calls and results from interactions (unordered set)
        pred_api_calls = []  # List[Tuple[name:str, args:dict, result:dict]]
        try:
            interactions = item.get('interactions', [])
            for inter in interactions:
                if inter.get('type') == 'tool_call':
                    # Parse tool_call_query
                    call_query_str = inter.get('tool_call_query', '')
                    try:
                        call_query = json.loads(call_query_str)
                        tool_name = call_query.get('name')
                        tool_args = call_query.get('arguments', {})
                    except Exception:
                        tool_name, tool_args = None, {}
                    # Parse tool_response
                    resp_raw = inter.get('tool_response')
                    try:
                        pred_result = json.loads(resp_raw) if isinstance(resp_raw, str) else resp_raw
                    except Exception:
                        pred_result = {}
                    pred_api_calls.append((tool_name, tool_args, pred_result))
        except Exception:
            pred_api_calls = []
        
        used_pred_indices = set()
        item_gt_api_calls = 0
        item_correct_api_calls = 0
        
        # Evaluate API call accuracy (unordered matching: each API in GT only needs to be correctly called by any prediction)
        for j, turn in enumerate(chat_history):
            if turn['role'] != 'API':
                continue
            total_api_calls += 1
            item_gt_api_calls += 1
            gt_api_name = turn['api_name']
            gt_input = turn['param_dict']
            gt_result = turn['result']
            
            # Candidates: unused predictions with same name
            candidate_indices = [idx for idx, (pname, _, _) in enumerate(pred_api_calls)
                                 if idx not in used_pred_indices and pname == gt_api_name]
            matched = False
            if candidate_indices:
                # Try each one, any passing counts as correct and consumes that prediction
                for idx in candidate_indices:
                    _, pred_args, _ = pred_api_calls[idx]
                    if isinstance(pred_args, dict) and pred_args == gt_input:
                        correct_api_calls += 1
                        item_correct_api_calls += 1
                        used_pred_indices.add(idx)
                        matched = True
                        break
                
                if not matched:
                    # Has same-name candidates but input mismatch, attributed to input mismatch
                    error_statistics['INPUT_MISMATCH']['count'] += 1
            else:
                # No same-name candidates: if no predictions at all then NO_API_CALL, otherwise name mismatch
                if len(pred_api_calls) == 0:
                    error_statistics['NO_API_CALL']['count'] += 1
                    error_statistics['NO_API_CALL']['samples'].append({
                        'file': file_name,
                        'sample_id': i,
                        'turn_id': j,
                        'ground_truth': turn,
                        'output': output
                    })
                else:
                    error_statistics['API_NAME_MISMATCH']['count'] += 1
        
        # Record detailed item (preserve original fields)
        item_record = dict(item)
        item_record['metrics'] = {
            'gt_api_calls': item_gt_api_calls,
            'correct_api_calls': item_correct_api_calls,
            'api_accuracy_item': (item_correct_api_calls / item_gt_api_calls) if item_gt_api_calls > 0 else 0.0,
            'success_rate': 1.0 if (item_correct_api_calls / item_gt_api_calls) == 1.0 and item_gt_api_calls > 0 else 0.0,
            'rouge_l': float(rouge_score),
        }
        item_record['pred_final_response'] = final_answer
        detailed_items.append(item_record)
        
    # Calculate overall metrics
    api_accuracy = correct_api_calls / total_api_calls if total_api_calls > 0 else 0
    success_rate = sum(item['metrics']['success_rate'] for item in detailed_items) / len(detailed_items) if detailed_items else 0.0
    avg_rouge_score = total_rouge_score / len(data) if len(data) > 0 else 0
    
    # Save detailed results (each sample + metrics)
    with open(os.path.join(output_dir, output_metrics_path), 'w', encoding='utf-8') as f:
        json.dump(detailed_items, f, indent=4, ensure_ascii=False)
    
    # Save overall metrics
    overall_metrics = {
        'api_accuracy': api_accuracy,
        'success_rate': success_rate,
        'avg_rouge_score': avg_rouge_score,
        'total_api_calls': total_api_calls,
        'correct_api_calls': correct_api_calls,
        'total_samples': len(data),
        'error_statistics': error_statistics
    }
    with open(os.path.join(output_dir, output_metrics_overall_path), 'w', encoding='utf-8') as f:
        json.dump(overall_metrics, f, indent=4, ensure_ascii=False)
    
    print(f"API Accuracy: {api_accuracy:.4f}")
    print(f"Success Rate: {success_rate:.4f}")
    print(f"Average Rouge-L Score: {avg_rouge_score:.4f}")
    print(f"Total API calls: {total_api_calls}")
    print(f"Correct API calls: {correct_api_calls}")
    
    return overall_metrics


def evaluate_api_bank_level3(
    data: List[Dict],
    output_list: List[str],
    output_dir: str,
    output_metrics_path: str,
    output_metrics_overall_path: str,
    lv3_apis_dir: str
):
    """Evaluate API-Bank Level-3 (scenario requiring tool search)"""
    
    print("Evaluating API-Bank Level-3...")
    
    # Initialize Level-3 tool executor (no longer executes)
    executor = APIBankExecutor(apis_dir=lv3_apis_dir)
    
    total_api_calls = 0
    correct_api_calls = 0
    total_rouge_score = 0
    detailed_items: List[Dict[str, Any]] = []
    
    error_statistics = {
        'NO_API_CALL': {'count': 0, 'samples': []},
        'API_NAME_MISMATCH': {'count': 0, 'samples': []},
        'HAS_EXCEPTION': {'count': 0, 'samples': []},
        'INPUT_MISMATCH': {'count': 0, 'samples': []},
        'OUTPUT_MISMATCH': {'count': 0, 'samples': []},
        'INVALID_INPUT_PARAMETER': {'count': 0, 'samples': []},
        'KEY_ERROR': {'count': 0, 'samples': []},
        'FAILED_PARSE_API_CALL': {'count': 0, 'samples': []},
        'MISS_INPUT_ARGUMENT': {'count': 0, 'samples': []},
    }
    
    for i, (item, output) in tqdm(enumerate(zip(data, output_list))):
        file_name = item.get('file', f'sample_{i}')
        requirement = item.get('requirement', '')
        gt_response = item.get('response', '')
        all_api_calls = item.get('apis', [])
        api_calls = []
        for api_call in all_api_calls:
            if api_call['api_name'] == 'ToolSearcher':
                continue
            api_calls.append(api_call)
        
        # Calculate Rouge-L score for final response
        rouge_score = 0.0
        if gt_response:
            rouge_score = calculate_rouge_l_score(gt_response, output)
            total_rouge_score += rouge_score
        
        # Extract executed API calls from interactions (unordered set)
        pred_api_calls = []  # List[Tuple[name:str, args:dict, result:dict]]
        try:
            interactions = item.get('interactions', [])
            for inter in interactions:
                if inter.get('type') == 'tool_call':
                    call_query_str = inter.get('tool_call_query', '')
                    try:
                        call_query = json.loads(call_query_str)
                        tool_name = call_query.get('name')
                        tool_args = call_query.get('arguments', {})
                    except Exception:
                        tool_name, tool_args = None, {}
                    resp_raw = inter.get('tool_response')
                    try:
                        pred_result = json.loads(resp_raw) if isinstance(resp_raw, str) else resp_raw
                    except Exception:
                        pred_result = {}
                    pred_api_calls.append((tool_name, tool_args, pred_result))
        except Exception:
            pred_api_calls = []
        
        used_pred_indices = set()
        item_gt_api_calls = 0
        item_correct_api_calls = 0
        
        # Process each API call (unordered matching)
        for j, api_call in enumerate(api_calls):
            total_api_calls += 1
            item_gt_api_calls += 1
            gt_api_name = api_call['api_name']
            gt_input = api_call.get('input', {})
            gt_result = api_call.get('output', {})
            
            candidate_indices = [idx for idx, (pname, _, _) in enumerate(pred_api_calls)
                                 if idx not in used_pred_indices and pname == gt_api_name]
            matched = False
            if candidate_indices:
                for idx in candidate_indices:
                    _, pred_args, _ = pred_api_calls[idx]
                    if isinstance(pred_args, dict) and pred_args == gt_input:
                        correct_api_calls += 1
                        item_correct_api_calls += 1
                        used_pred_indices.add(idx)
                        matched = True
                        break
                if not matched:
                    error_statistics['INPUT_MISMATCH']['count'] += 1
            else:
                if len(pred_api_calls) == 0:
                    error_statistics['NO_API_CALL']['count'] += 1
                else:
                    error_statistics['API_NAME_MISMATCH']['count'] += 1
        
        # Record detailed item (preserve original fields)
        item_record = dict(item)
        item_record['metrics'] = {
            'gt_api_calls': item_gt_api_calls,
            'correct_api_calls': item_correct_api_calls,
            'api_accuracy_item': (item_correct_api_calls / item_gt_api_calls) if item_gt_api_calls > 0 else 0.0,
            'success_rate': 1.0 if (item_correct_api_calls / item_gt_api_calls) == 1.0 and item_gt_api_calls > 0 else 0.0,
            'rouge_l': float(rouge_score),
        }
        item_record['pred_final_response'] = output
        detailed_items.append(item_record)
    
    # Calculate overall metrics
    api_accuracy = correct_api_calls / total_api_calls if total_api_calls > 0 else 0
    success_rate = sum(item['metrics']['success_rate'] for item in detailed_items) / len(detailed_items) if detailed_items else 0.0
    avg_rouge_score = total_rouge_score / len(data) if len(data) > 0 else 0
    
    # Save detailed results (each sample + metrics)
    with open(os.path.join(output_dir, output_metrics_path), 'w', encoding='utf-8') as f:
        json.dump(detailed_items, f, indent=4, ensure_ascii=False)
    
    # Save overall metrics
    overall_metrics = {
        'api_accuracy': api_accuracy,
        'success_rate': success_rate,
        'avg_rouge_score': avg_rouge_score,
        'total_api_calls': total_api_calls,
        'correct_api_calls': correct_api_calls,
        'total_samples': len(data),
        'error_statistics': error_statistics
    }
    with open(os.path.join(output_dir, output_metrics_overall_path), 'w', encoding='utf-8') as f:
        json.dump(overall_metrics, f, indent=4, ensure_ascii=False)
    
    print(f"API Accuracy: {api_accuracy:.4f}")
    print(f"Success Rate: {success_rate:.4f}")
    print(f"Average Rouge-L Score: {avg_rouge_score:.4f}")
    print(f"Total API calls: {total_api_calls}")
    print(f"Correct API calls: {correct_api_calls}")
    
    return overall_metrics


def evaluate_api_bank_predictions(
    data: List[Dict],
    output_list: List[str],
    output_dir: str,
    output_metrics_path: str,
    output_metrics_overall_path: str,
    args
):
    """Evaluate API-Bank prediction results"""
    
    # Choose evaluation method based on dataset type
    if not args.enable_tool_search:
        return evaluate_api_bank_level1(
            data=data,
            output_list=output_list,
            output_dir=output_dir,
            output_metrics_path=output_metrics_path,
            output_metrics_overall_path=output_metrics_overall_path,
            apis_dir=args.api_bank_apis_dir
        )
    else:
        return evaluate_api_bank_level3(
            data=data,
            output_list=output_list,
            output_dir=output_dir,
            output_metrics_path=output_metrics_path,
            output_metrics_overall_path=output_metrics_overall_path,
            lv3_apis_dir=args.api_bank_lv3_apis_dir
        )
