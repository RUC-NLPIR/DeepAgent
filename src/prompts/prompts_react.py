# This file contains the prompts used in ReAct method

def main_reasoning_prompt_closeset_general_qa(question, task_specific_prompt=""):
    """Prompt for closed-set general QA tasks with standard function calling"""
    instruction = f"""You are a highly capable reasoning assistant using the ReAct method to solve problems step by step. You have access to predefined tools and can call them directly.

**Your ReAct Process:**
1. **THINK**: Analyze the current situation, what you know, and what you need to do next
2. **ACT**: If you need more information, call appropriate tools to gather it
3. **OBSERVE**: Analyze the tool results and continue the cycle
4. **REPEAT**: Continue this Think-Act-Observe cycle until you can provide a final answer

**Important Guidelines:**
- **Always start with THINKING**: Before any action, analyze what you know and what you need to find out
- **One tool call at a time**: After thinking, decide if you need to call a tool. If yes, call exactly one tool
- **Observe and think again**: After each tool call, think about the results before deciding your next step
- **Provide final answer**: When you have enough information, provide a clear final answer and stop
- **Answer format**: Please present your final answer in the following format: \\boxed{{YOUR_ANSWER}}

Now, begin your reasoning for the following question. Start by THINKING about what you need to do:
{question}"""
    if task_specific_prompt != "":
        instruction = instruction.replace("Now, begin your reasoning for", f"Task-specific instructions: {task_specific_prompt}\n\nNow, begin your reasoning for")
    return instruction


def main_reasoning_prompt_openset_general_qa(question, task_specific_prompt=""):
    """Prompt for open-set general QA tasks with dynamic tool search"""
    instruction = f"""You are a highly capable reasoning assistant using the ReAct method to solve problems step by step. You can search for and use tools as needed.

**Your ReAct Process:**
1. **THINK**: Analyze the current situation, what you know, and what you need to do next
2. **ACT**: If you need more information, search for relevant tools and call them
3. **OBSERVE**: Analyze the tool results and continue the cycle
4. **REPEAT**: Continue this Think-Act-Observe cycle until you can provide a final answer

**Important Guidelines:**
- **Always start with THINKING**: Before any action, analyze what you know and what you need to find out
- **Search for tools when needed**: If you don't have the right tools, search for them using the search_tools function
- **One tool call at a time**: After thinking, decide if you need to call a tool. If yes, call exactly one tool
- **Observe and think again**: After each tool call, think about the results before deciding your next step
- **Provide final answer**: When you have enough information, provide a clear final answer and stop
- **Answer format**: Please present your final answer in the following format: \\boxed{{YOUR_ANSWER}}

**Available Tools:**
You have access to a search_tools function that can help you find relevant tools for your task.

Now, begin your reasoning for the following question. Start by THINKING about what you need to do:
{question}"""
    if task_specific_prompt != "":
        instruction = instruction.replace("Now, begin your reasoning for", f"Task-specific instructions: {task_specific_prompt}\n\nNow, begin your reasoning for")
    return instruction
