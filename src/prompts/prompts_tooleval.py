# ToolBench Evaluation Prompts
# Based on the original ToolBench evaluator prompts

CHECK_ANSWER_STATUS_PROMPT = """You are an evaluation assistant. Please determine if the answer correctly addresses the query.

Query: {query}

Answer: {answer}

Please analyze if the answer is correct, incorrect, or unclear. Consider:
1. Does the answer directly address the query?
2. Is the answer accurate and complete?
3. Does the answer provide the requested information or solution?

Respond with one of the following:
- "Solved": The answer correctly and completely addresses the query
- "Unsolved": The answer is incorrect or does not address the query
- "Unsure": The answer is unclear or partially addresses the query

Please provide your assessment:"""

PARSE_ANSWER_STATUS_PROMPT = """You are an evaluation assistant. Please analyze the detailed answer structure to determine if the task was solved.

Query: {query}

Answer Details: {answer}

Please examine the answer structure, tool usage, and final response to determine if the task was successfully completed.

Consider:
1. Were appropriate tools used correctly?
2. Did the agent follow a logical reasoning process?
3. Was a final answer provided that addresses the query?
4. Were there any errors or incomplete steps?

Respond with one of the following:
- "Solved": The task was successfully completed with appropriate tool usage and a correct final answer
- "Unsolved": The task was not completed due to errors, incorrect tool usage, or missing final answer
- "Unsure": The completion status is unclear due to partial information or ambiguous results

Please provide your assessment:"""

CHECK_TASK_SOLVABLE_PROMPT = """You are an evaluation assistant. Please determine if the given task is solvable with the available tools.

Task: {task}

Please analyze:
1. Is the task clearly defined and understandable?
2. Are the required tools available to complete this task?
3. Is the task within the scope of what can be accomplished with the given toolset?
4. Are there any fundamental limitations that would prevent task completion?

Respond with one of the following:
- "Solvable": The task can be completed with the available tools
- "Unsolvable": The task cannot be completed due to missing tools or fundamental limitations
- "Unsure": It is unclear whether the task can be completed

Please provide your assessment:"""

SELECT_BETTER_ANSWER_PROMPT = """You are an evaluation assistant. Please compare two answers to the same query and select the better one.

Query: {query}

Answer 0: {answer_0}

Answer 1: {answer_1}

Please compare these answers based on:
1. Completeness: Which answer more thoroughly addresses the query?
2. Accuracy: Which answer provides more correct information?
3. Tool Usage: Which answer demonstrates better tool selection and usage?
4. Reasoning: Which answer shows clearer logical thinking?
5. Final Result: Which answer provides a better final solution?

Respond with the index of the better answer (0 or 1) and a brief explanation of your choice.

Format your response as:
Index: [0 or 1]
Reason: [brief explanation]""" 