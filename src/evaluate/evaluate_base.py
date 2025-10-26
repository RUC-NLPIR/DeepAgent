import re
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
import string
import os, time
from collections import defaultdict
# from lcb_runner.evaluation import codegen_metrics
import sys
sys.path.append('./src')
from utils.math_equivalence import is_equiv
from openai import OpenAI, AsyncOpenAI
import asyncio
from typing import List
from prompts.prompts_deepagent import BEGIN_TOOL_RESPONSE, END_TOOL_RESPONSE


def extract_answer_fn(output, mode='qa', extract_answer=False):
    if extract_answer == False and mode not in ['infogen', 'summary', 'research']:
        if mode == 'qa':
            return output.strip()
        pred_answer_lines = output.replace("\n\n", "\n").strip().split('\n')
        pred_answer = '\n'.join(pred_answer_lines[-3:])
        return pred_answer
    extracted_text = ''
    if mode == 'codegen':
        pattern = r'```python\s*(.*?)\s*```'  # Extract the code between ```python and ```
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()  # Take the last match
    elif mode in ['infogen', 'summary', 'research']:
        pattern_info = "**Final Information"
        if "</think>\n" in output:
            extracted_text = output.split("</think>\n")[-1].split("<|begin_click_link|>")[0].replace(pattern_info, "").strip(':**').strip('\n').strip("```").strip()  # Extract content after </think>
            if mode == 'infogen':
                extracted_text = '\n'.join(extracted_text.replace("\n\n", "\n").split('\n')[:5])  # Keep only the first 5 lines
        elif pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].split("<|begin_click_link|>")[0].strip('\n').strip(':**').strip("```").strip()  # Extract content after **Final Information**
            if mode == 'infogen':
                extracted_text = '\n'.join(extracted_text.replace("\n\n", "\n").split('\n')[:5])  # Keep only the first 5 lines
        else:
            # extracted_text = "No helpful information found."
            extracted_text = '\n'.join(output.strip().replace("</think>\n", "").replace("\n\n", "\n").split('\n')[-5:])  # If not extracted, keep only the last 5 lines
        if mode == 'research':
            extracted_text = extracted_text[:6000]
        else:
            extracted_text = extracted_text[:2500]
    elif mode in ['math', 'choose', 'qa', 'toolhop']:
        if mode =='toolhop':
            pattern = r'\\boxed\{(.*?)\}'
        else:
            pattern = r'\\boxed{(.*)}'
        matches = re.findall(pattern, output, re.DOTALL)
        if matches:
            extracted_text = matches[-1]  # Take the last match
        else:
            pattern = 'ANSWER:'
            if pattern in output:
                extracted_text = output.split(pattern)[-1].strip('**').strip()
        if mode in ['choose']:
            inner_pattern = r'\\text\{(.*?)\}'
            inner_matches = re.findall(inner_pattern, extracted_text)
            if inner_matches:
                extracted_text = inner_matches[-1]  # Take the last match
            extracted_text = extracted_text.strip("()")
    return extracted_text


async def llm_evaluate_equivalence_single(
    client: AsyncOpenAI,
    question: str,
    labeled_answer: str,
    pred_answer: str,
    model_name: str,
    semaphore: asyncio.Semaphore,
    retry_limit: int = 3,
    extract_answer: bool = False,
) -> bool:
    """Evaluate a single pair of answers using LLM"""

    if extract_answer:
        prompt = f"""You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer. If the answers are expressed differently but are mathematically or semantically equivalent, they should be considered correct.

Question: {question}

Labeled Answer: {labeled_answer}

Predicted Answer: {pred_answer}

Are these answers equivalent? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""
    else:
        prompt = f"""You are an evaluation assistant. Please determine if the model output is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {labeled_answer}

Model Output (Last few lines): {pred_answer}

Did the model give an answer equivalent to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""

    for attempt in range(retry_limit):
        try:
            async with semaphore:
                chat_response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = chat_response.choices[0].message.content.strip()
                llm_judge = is_equiv(pred_answer, labeled_answer) or \
                    response_text.lower() == "correct" and \
                    not ("incorrect" in response_text.lower() or \
                         "wrong" in response_text.lower() or \
                         "not correct" in response_text.lower())
                return llm_judge, response_text
        except Exception as e:
            if attempt == retry_limit - 1:
                print(f"Error in LLM evaluation: {e}")
                return is_equiv(pred_answer, labeled_answer), "Error"
            await asyncio.sleep(1 * (attempt + 1))
    
    return is_equiv(pred_answer, labeled_answer), "Error"


async def llm_evaluate_equivalence_batch(
    questions: List[str],
    labeled_answers: List[str], 
    pred_answers: List[str],
    model_name: str = None,
    client: AsyncOpenAI = None,
    semaphore: asyncio.Semaphore = None,
    extract_answer: bool = False
) -> List[bool]:
    """
    Evaluate multiple answer pairs concurrently using LLM
    """

    tasks = [
        llm_evaluate_equivalence_single(
            client=client,
            question=q,
            labeled_answer=l,
            pred_answer=p,
            model_name=model_name,
            semaphore=semaphore,
            extract_answer=extract_answer
        )
        for q, l, p in zip(questions, labeled_answers, pred_answers)
    ]

    with tqdm(total=len(tasks), desc="LLM Evaluation") as pbar:
        async def track_progress(task):
            result = await task
            pbar.update(1)
            return result
            
        tracked_tasks = [track_progress(task) for task in tasks]
        results = await asyncio.gather(*tracked_tasks)
    
    return results


def evaluate_predictions(output, labeled_answer, mode='math', use_llm=False, question=None, extract_answer=False):
    final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0, 'llm_equal': 0}
    pred_answer = extract_answer_fn(output, mode=mode, extract_answer=extract_answer)
    pred_answer_new = "\\boxed{" + pred_answer + "}"  # Take the last match
    if pred_answer != '':
        final_metric["is_valid_answer"] = True
    else:
        # If no answer was extracted, keep only the last 5 lines
        pred_answer_new = '\n'.join(output.replace("\n\n", "\n").strip().split('\n')[-5:])

    if mode in ['qa']:
        def normalize_answer_qa(s):
            def remove_articles(text):
                return re.sub(r"\b(a|an|the)\b", " ", text)
            def white_space_fix(text):
                return " ".join(text.strip().split())
            def remove_punc(text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)
            def lower(text):
                return text.lower()
            return white_space_fix(remove_articles(remove_punc(lower(s))))
        normalized_pred_answer = normalize_answer_qa(pred_answer_new)

        for answer in labeled_answer:
            normalized_ground_truth = normalize_answer_qa(answer)
            em = int(normalized_pred_answer == normalized_ground_truth)
            acc = int(normalized_ground_truth in normalized_pred_answer)

            prediction_tokens = normalized_pred_answer.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["em", "acc", "f1"]:
                final_metric[k] = max(eval(k), final_metric[k])

    elif mode in ['math', 'choose']:
        def normalize_answer(text):
            text = text.lower()
            text = " ".join(text.strip().split())
            return text
        normalized_pred_answer = normalize_answer(pred_answer_new)
        normalized_ground_truth = normalize_answer(labeled_answer)

        em = int(normalized_pred_answer == normalized_ground_truth)
        acc = int(normalized_ground_truth in normalized_pred_answer)
    
        prediction_tokens = normalized_pred_answer.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
            recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
            if (precision + recall) == 0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)

        final_metric["em"] = em
        final_metric["acc"] = acc
        final_metric["f1"] = f1

        final_metric["math_equal"] = is_equiv(normalized_pred_answer, normalized_ground_truth)
        
        # Add LLM-based evaluation if requested
        if use_llm and question is not None:
            final_metric["llm_equal"] = 0  # Will be updated in batch later

    return final_metric, pred_answer


async def run_evaluation(data, input_list, output_list, output_dir, output_metrics_path, output_metrics_overall_path, use_llm=False, extract_answer=False, domain_fields=None, client=None, semaphore=None, model_name=None):
    # Initialize domain metrics dictionary
    domain_metrics = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'em': [],
        'acc': [],
        'f1': [],
        'math_equal': [],
        'llm_equal': [],
        'pass@1': []
    })

    # Helper function to get domain from item
    def get_all_domains(item):
        all_domains = set()
        for field in domain_fields:
            if field in item and item[field] is not None:
                all_domains.add(item[field])
        return all_domains

    # Evaluation for math/qa tasks
    avg_em, avg_acc, avg_f1, avg_math, avg_llm = [], [], [], [], []
    num_valid_answer = 0
    
    # Lists to store data for batch LLM evaluation
    questions_for_llm = []
    labeled_answers_for_llm = []
    pred_answers_for_llm = []
    items_for_llm = []

    for item, input_prompt, result in tqdm(zip(data, input_list, output_list), total=len(input_list)):
        if result == '':
            item['Pred_Answer'] = ''
            item['Metrics'] = {
                'em': 0,
                'acc': 0,
                'f1': 0,
                'math_equal': 0,
                'llm_equal': 0 if use_llm else None
            }
            avg_em.append(0)
            avg_acc.append(0)
            avg_f1.append(0)
            avg_math.append(0)
            if use_llm:
                avg_llm.append(0)
            continue

        # Get the labeled answer from the item
        labeled_answer = item.get('answer', '')  # Use get() to safely access the answer field
        if 'Correct Choice' in item and item['Correct Choice'] is not None:
            labeled_answer = item['Correct Choice']
        elif 'answer_letter' in item and item['answer_letter'] is not None:
            labeled_answer = item['answer_letter']
        metric, pred_answer = evaluate_predictions(
            output=result, 
            labeled_answer=labeled_answer,
            use_llm=use_llm,
            question=input_prompt,
            extract_answer=extract_answer
        )
        
        item['Pred_Answer'] = pred_answer
        item['Metrics'] = metric

        # Store data for batch LLM evaluation
        if use_llm:
            questions_for_llm.append(input_prompt)
            labeled_answers_for_llm.append(labeled_answer)
            pred_answers_for_llm.append(pred_answer)
            items_for_llm.append(item)

        # Determine the validity of the predicted answer
        my_method_valid = (pred_answer != '')

        avg_em.append(metric['em'])
        avg_acc.append(metric['acc'])
        avg_f1.append(metric['f1'])
        avg_math.append(metric['math_equal'])

        if my_method_valid:
            num_valid_answer += 1

    # Perform batch LLM evaluation if needed
    if use_llm and questions_for_llm:
        llm_results = await llm_evaluate_equivalence_batch(
            questions=questions_for_llm,
            labeled_answers=labeled_answers_for_llm,
            pred_answers=pred_answers_for_llm,
            extract_answer=extract_answer,
            client=client,
            semaphore=semaphore,
            model_name=model_name
        )
        
        # Update metrics with LLM results
        for item, (llm_result, llm_response) in zip(items_for_llm, llm_results):
            item['Metrics']['llm_equal'] = int(llm_result)
            item['Metrics']['llm_response'] = llm_response
            avg_llm.append(int(llm_result))

    # Compute overall metrics
    overall_metrics = {
        'em': float(np.mean(avg_em)) if len(avg_em) > 0 else 0.0,
        'acc': float(np.mean(avg_acc)) if len(avg_acc) > 0 else 0.0,
        'f1': float(np.mean(avg_f1)) if len(avg_f1) > 0 else 0.0,
        'math_equal': float(np.mean(avg_math)) if len(avg_math) > 0 else 0.0,
        'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
    }
    
    # Add LLM evaluation metric if available
    if len(avg_llm) > 0:
        overall_metrics['llm_equal'] = float(np.mean(avg_llm))

    for item, metric in zip(data, [item['Metrics'] for item in data]):
        all_domains = get_all_domains(item)
        for domain in all_domains:
            domain_metrics[domain]['total'] += 1
            domain_metrics[domain]['em'].append(metric['em'])
            domain_metrics[domain]['acc'].append(metric['acc'])
            domain_metrics[domain]['f1'].append(metric['f1'])
            domain_metrics[domain]['math_equal'].append(metric['math_equal'])
            if 'llm_equal' in metric:
                domain_metrics[domain]['llm_equal'].append(metric['llm_equal'])

    # After the main evaluation loop and before saving metrics, add:
    # Calculate domain-specific metrics
    domain_metrics_final = {}
    for domain, metrics in domain_metrics.items():
        domain_metrics_final[domain] = {
            'total': metrics['total'],
            'em': float(np.mean(metrics['em'])) if len(metrics['em']) > 0 else 0.0,
            'acc': float(np.mean(metrics['acc'])) if len(metrics['acc']) > 0 else 0.0,
            'f1': float(np.mean(metrics['f1'])) if len(metrics['f1']) > 0 else 0.0,
            'math_equal': float(np.mean(metrics['math_equal'])) if len(metrics['math_equal']) > 0 else 0.0,
        }
        if metrics['llm_equal']:
            domain_metrics_final[domain]['llm_equal'] = float(np.mean(metrics['llm_equal']))
        if metrics['pass@1']:
            domain_metrics_final[domain]['pass@1'] = float(np.mean(metrics['pass@1']))

    # Add domain metrics to overall metrics
    overall_metrics['domain_metrics'] = domain_metrics_final
    
    print('---'*10, '\n', overall_metrics)
    
    # Save prediction results and metrics
    with open(os.path.join(output_dir, output_metrics_path), mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, output_metrics_overall_path), mode='w', encoding='utf-8') as json_file:
        json.dump(overall_metrics, json_file, indent=4, ensure_ascii=False)


def evaluate_predictions_toolhop(data, output_list, output_dir, output_metrics_path, output_metrics_overall_path):
    all_answer_corrects = []
    all_path_scores = []
    all_has_valid_answer = []
    all_action_counts = []

    for item, result in zip(data, output_list):
        if type(result) == str:
            output = result
        else:
            output = result.outputs[0].text

        # (1) answer_correct
        pred_answer = extract_answer_fn(output, mode='toolhop', extract_answer=True)
        
        has_valid_answer = 1 if pred_answer.strip() else 0
        all_has_valid_answer.append(has_valid_answer)
        
        labeled_answer = str(item.get('answer', '')).strip().lower()
        normalized_pred_answer = pred_answer.strip().lower()
        
        answer_correct = 1 if normalized_pred_answer == labeled_answer else 0
        all_answer_corrects.append(answer_correct)
        
        # (2) path_score
        subtask_answers = set(str(v).strip().lower() for v in item.get('sub_task', {}).values())
        
        tool_response_pattern = rf"{BEGIN_TOOL_RESPONSE}(.*?){END_TOOL_RESPONSE}"
        tool_responses_str = re.findall(tool_response_pattern, output, re.DOTALL)
        
        solved_subtask_answers = set()
        for subtask_answer in subtask_answers:
            for res_str in tool_responses_str:
                if subtask_answer in res_str.lower():
                    solved_subtask_answers.add(subtask_answer)
                    break  # Only need to appear in any tool_response to be considered solved

        path_score = 0.0
        if subtask_answers:
            path_score = len(solved_subtask_answers) / len(subtask_answers)
        
        all_path_scores.append(path_score)
        
        action_count = item.get('action_count', 0)
        all_action_counts.append(action_count)
        
        item['pred_answer'] = normalized_pred_answer
        item['metrics'] = {
            'has_valid_answer': has_valid_answer,
            'answer_correct': answer_correct,
            'path_score': path_score,
        }

    overall_metrics = {
        'total_instance': len(data),
        'has_valid_answer': float(np.mean(all_has_valid_answer)) if all_has_valid_answer else 0.0,
        'average_actions': float(np.mean(all_action_counts)) if all_action_counts else 0.0,
        'answer_correct': float(np.mean(all_answer_corrects)) if all_answer_corrects else 0.0,
        'path_score': float(np.mean(all_path_scores)) if all_path_scores else 0.0,
    }
    
    print("ToolHop Evaluation Metrics:")
    print(overall_metrics)
    
    # Save prediction results and metrics
    with open(os.path.join(output_dir, output_metrics_path), mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, output_metrics_overall_path), mode='w', encoding='utf-8') as json_file:
        json.dump(overall_metrics, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model outputs.")
    parser.add_argument('--output_path', type=str, required=True, help='Path to the model output JSON file.')
    parser.add_argument('--task', type=str, required=True, choices=['code', 'math', 'choose', 'qa', 'llm', 'toolhop'], help='Task type for evaluation')
    parser.add_argument('--use_llm', action='store_true', help='Use LLM for equivalence evaluation')
    parser.add_argument('--extract_answer', action='store_true', help='Extract answer from output')
    parser.add_argument('--api_base_url', type=str, default='http://10.148.6.244:8083/v1', help='Base URL for LLM API')
    parser.add_argument('--model_name', type=str, default='qwen2.5-72b-instruct', help='Model name for LLM evaluation')
    parser.add_argument('--api_key', type=str, default='empty', help='API key for LLM API')
    args = parser.parse_args()

    # Define the list of domain field names to check (in order of priority)
    DOMAIN_FIELDS = ['Level', 'level', 'category', 'High-level domain', 'difficulty_level', 'field', 'problem_type']

    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    base_name = os.path.basename(output_path)
    output_metrics_path = base_name.replace('.json', '.metrics.json')
    output_metrics_overall_path = base_name.replace('.json', '.metrics.overall.json')

    # Load main output data
    with open(output_path, mode='r', encoding='utf-8') as file:
        data = json.load(file)

    if args.task == 'toolhop':
        data = data
        output_list = [item.get('output', item.get('result')) for item in data]
        evaluate_predictions_toolhop(
            data=data,
            output_list=output_list,
            output_dir=output_dir,
            output_metrics_path=output_metrics_path,
            output_metrics_overall_path=output_metrics_overall_path,
        )
    else:
        # Prepare input_list and output_list for run_evaluation
        input_list = []
        output_list = []
        
        if isinstance(data, dict):
            # Convert dict to list if data is a dictionary
            data = []
            for key, item in data.items():
                if isinstance(item, dict):  # Ensure item is a dictionary
                    data.append(item)
                    input_list.append(item.get('question'))
                    output_list.append(item.get('result'))
        else:
            # If data is already a list
            data = data
            input_list = [item.get('Question', item.get('question')) for item in data]
            output_list = [item.get('Output', item.get('output')) for item in data]

        # Initialize LLM client and semaphore if needed
        llm_client = None
        llm_semaphore = None
        if args.use_llm:
            llm_client = AsyncOpenAI(base_url=args.api_base_url, api_key=args.api_key)
            llm_semaphore = asyncio.Semaphore(50)

        # Run evaluation with domain fields
        asyncio.run(run_evaluation(
            data=data,  # Pass the properly structured data
            input_list=input_list,
            output_list=output_list,
            output_dir=output_dir,
            output_metrics_path=output_metrics_path,
            output_metrics_overall_path=output_metrics_overall_path,
            use_llm=args.use_llm,
            extract_answer=args.extract_answer,
            domain_fields=DOMAIN_FIELDS,  # Pass the domain fields to run_evaluation
            client=llm_client,
            semaphore=llm_semaphore,
            model_name=args.model_name,
        ))

    print(f"Evaluation completed. Metrics saved to {os.path.join(output_dir, output_metrics_path)}")
