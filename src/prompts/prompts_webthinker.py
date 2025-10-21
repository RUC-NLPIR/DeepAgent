

def get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )



def get_deep_web_explorer_instruction(search_query, search_intent, search_result):
    return f"""You are a web explorer analyzing search results to find relevant information based on a given search query and search intent.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **More Information Seeking:**
- If the information is not relevant to the query, you could:
  1. Search again: <|begin_search_query|>another search query<|end_search_query|>
  2. Access webpage content using: <|begin_click_link|>your URL<|end_click_link|>

3. **Extract Relevant Information:**
- Return the relevant information from the **Searched Web Pages** that is relevant to the **Current Search Query**.

4. **Output Format:**
- Present the information beginning with **Final Information** as shown below.

**Final Information**
[Relevant information]

**Inputs:**

- **Current Search Query:**
{search_query}

- **Detailed Search Intent:**
{search_intent}

- **Searched Web Pages:**
{search_result}

Now please analyze the web pages and extract relevant information for the search query "{search_query}" and the search intent.
"""




def get_web_page_reader_instruction(query, document):
    return f"""{document}
Please provide all content related to "{query}" from this document in markdown format.
If there isn't any relevant information, just output "No relevant information". If there is any relevant information, output all the relevant information with potential helpful links."""

def get_detailed_web_page_reader_instruction(query, search_intent, document):
    return f"""Please provide all content related to the following search query and search intent from this document in markdown format.

Search Query: 
{query}

Search Intent: 
{search_intent}

Searched Web Page:
{document}

Instructions:
- Extract all content that matches the search query and intent, do not omit any relevant information.
- Include any relevant links from the source
- If no relevant information exists, output "No relevant information"
- Focus on factual, accurate information that directly addresses the query/intent
"""


def get_search_intent_instruction(prev_reasoning):
    return f"""Based on the previous thoughts below, provide the detailed intent of the latest search query.
Previous thoughts: {prev_reasoning}
Please provide the current search intent."""


def get_click_intent_instruction(prev_reasoning):
    return f"""Based on the previous thoughts below, provide the detailed intent of the latest click action.
Previous thoughts: {prev_reasoning}
Please provide the current click intent."""



def get_query_plan_instruction(question):
    return f"""You are a reasoning assistant. Your task is to generate a detailed query plan for answering the user's question by breaking it down into sub-queries.

Question: {question}

Please analyze the question and break it down into multiple sub-queries that will help gather all the necessary information to answer it completely. 

Output your query plan in JSON format as follows:

```json
{{
    "query_plan": [
        "sub-query-1",
        "sub-query-2",
        ...
    ]
}}
```
"""



def get_naive_rag_instruction(question, documents):
    return (
        "You are a knowledgeable assistant that uses the provided documents to answer the user's question.\n\n"
        "Question:\n"
        f"{question}\n"
        "Documents:\n"
        f"{documents}\n"
    )



def get_task_instruction_openqa(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'dpsk':
        user_prompt = (
            'Please answer the following question.\n\n'
            'Provide your final answer in the format **ANSWER: {YOUR_ANSWER}**.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_multi_choice(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following multiple-choice question. '
            'You should provide your final choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'dpsk':
        user_prompt = (
            'Please answer the following multiple-choice question.\n\n'
            'Provide your final choice in the format **ANSWER: {YOUR_CHOICE}**.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'llama':
        user_prompt = (
            'Please answer the following multiple-choice question. You should think step by step to solve it.\n\n'
            'Provide your final choice in the format \\boxed{YOUR_CHOICE}. Your final choice should be one of the letters A, B, C, or D, DO NOT include any answer content.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following multiple-choice question. You should think step by step to solve it.\n\n'
            'Provide your final choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt
