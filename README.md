<h1 align="center"> <img src="./figures/logo_deepagent.png" width="270" style="vertical-align:middle;"/><br>A General Reasoning Agent with Scalable Toolsets</a></h1>

<div align="center"> 

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2510.21618)
[![Paper](https://img.shields.io/badge/Paper-Hugging%20Face-orange?logo=huggingface)](https://huggingface.co/papers/2510.21618)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-yellow?logo=huggingface)](https://huggingface.co/datasets/lixiaoxi45/DeepAgent-Datasets)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](https://opensource.org/licenses/MIT) 
[![Python 3.10+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FXiaoxiLi0111%2Fstatus%2F1982649697467859438)](https://x.com/XiaoxiLi0111/status/1982649697467859438)
</div>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&size=20&duration=3000&pause=1000&color=005DE3&center=true&vCenter=true&width=800&lines=Welcome+to+DeepAgent;A+General+Reasoning+Agent+with+Scalable+Toolsets;Powered+by+RUC+x+Xiaohongshu+Inc." alt="Typing Animation" />
</div>


## 📣 Latest News

- **[October 27, 2025]**: 📄 Our paper is now available on **[arXiv](https://arxiv.org/abs/2510.21618)** and **[Hugging Face](https://huggingface.co/papers/2510.21618)**.
- **[October 27, 2025]**: 🚀 Our codebase released. You can now deploy DeepAgent with reasoning models like [QwQ](https://huggingface.co/collections/Qwen/qwq), [Qwen3](https://huggingface.co/collections/Qwen/qwen3) and your own toolsets.



## 🎬 Demo

<details open>
<summary><h3>1. General Agent Task with 16,000+ RapidAPIs</h3></summary>

<div align="center">
    <video src="https://github.com/user-attachments/assets/7aa586e9-a47a-425d-8d41-99226d2f6835" />
</div>

**DeepAgent** is a reasoning agent with scalable toolsets, capable of tackling general tasks by searching for and using the appropriate tools from over 16,000 RapidAPIs in an end-to-end agentic reasoning process. *(Note: Due to some APIs in ToolBench being unavailable, API responses are LLM-simulated in this demo to show the system's normal functionality.)*

</details>

<details open>
<summary><h3>2. Embodied AI Agent Task in ALFWorld Env.</h3></summary>

<div align="center">
  <video src="https://github.com/user-attachments/assets/fe309384-9102-4d1e-a929-f8b9b4041243" />
</div>

**DeepAgent** also excels at navigation-based tasks (e.g., web browsing, OS interaction, and embodied AI) by using a versatile set of pluggable actions such as moving, looking, and taking.
</details>

<details open>
<summary><h3>3. Deep Research Task with Specialized Tools</h3></summary>

<div align="center">
  <video src="https://github.com/user-attachments/assets/a6278bfd-2ee9-44aa-9f74-82aa826d8778" />
</div>

**DeepAgent** can also serve as a powerful research assistant, equipped with specialized tools for web search, browsing, code execution, visual QA, and file processing.
</details>




## 💡 Overview


<!-- ![Model Comparison](<./figures/comparison.png>) -->

**DeepAgent** is an end-to-end deep reasoning agent that performs autonomous thinking, tool discovery, and action execution within a single, coherent reasoning process. This paradigm shifts away from traditional, predefined workflows (e.g., ReAct's "Reason-Act-Observe" cycle), allowing the agent to maintain a global perspective on the entire task and dynamically discover tools on an as-needed basis.

To handle long-horizon interactions and prevent getting stuck in incorrect exploration paths, we introduce an **Autonomous Memory Folding** mechanism. This allows DeepAgent to "take a breath" by compressing its interaction history into a structured, brain-inspired memory schema, enabling it to reconsider its strategy and proceed efficiently.

Furthermore, we propose **ToolPO**, an end-to-end reinforcement learning (RL) training method tailored for general tool use, which enhances the agent's proficiency in mastering these complex mechanisms.

### 📊 Overall Performance

<div align="center">
  <img src="./figures/overall_results.png" width="70%" />
</div>

We conduct extensive experiments on a wide range of benchmarks:
- **(1) General Tool-Use Tasks:** We evaluate DeepAgent on ToolBench, API-Bank, TMDB, Spotify, and ToolHop, which feature toolsets scaling from tens to over ten thousand distinct tools. 
- **(2) Downstream Applications:** We test its performance on ALFWorld, WebShop, GAIA, and Humanity's Last Exam (HLE), which require the use of domain-specific toolsets. The overall results in Figure show that DeepAgent achieves superior performance across all scenarios.

### ✨ The DeepAgent Framework

![Framework](<./figures/framework.png>)
**Key Features:**

- **Unified Agentic Reasoning**: DeepAgent departs from rigid, predefined workflows. It operates in a single stream of thought, autonomously reasoning about the task, dynamically discovering necessary tools, and executing actions. This allows the LRM to maintain a global perspective and unlock its full autonomous potential.

- **Autonomous Memory Folding & Brain-Inspired Memory**: When facing complex problems, DeepAgent can autonomously trigger memory folding. This process consolidates the interaction history into a structured memory, allowing the agent to restart its reasoning with a condensed yet comprehensive understanding of its progress. The memory architecture is brain-inspired and consists of:
    - **Episodic Memory**: A high-level log of key events, decisions, and sub-task completions.
    - **Working Memory**: Contains the most recent information, including the current sub-goal and near-term plans.
    - **Tool Memory**: Consolidates tool-related interactions, allowing the agent to learn from experience and refine its strategies.

- **End-to-End RL Training with ToolPO**: To effectively train the agent, we introduce ToolPO, a policy optimization method featuring:
    - An **LLM-based Tool Simulator** that mimics real-world APIs, ensuring stable and efficient training.
    - **Tool-Call Advantage Attribution**, which assigns fine-grained credit to correct tool invocation tokens, providing a more precise learning signal.


## 🔧 Installation

###  Environment Setup
```bash
# Create conda environment
conda create -n deepagent python=3.10
conda activate deepagent

# Install requirements
cd DeepAgent-main
pip install -r requirements.txt
```

  

<details open>
<summary><h3>📊 Benchmarks</h3></summary>

The benchmarks we utilize are categorized into several types:
- **General Tool Use Benchmarks:**
    - [ToolBench](https://arxiv.org/abs/2307.16789): Features 16,000+ real-world RapidAPIs requiring multi-step, multi-tool reasoning.
    - [API-Bank](https://arxiv.org/abs/2304.08244): Evaluates planning, retrieval, and calling with 73 APIs across 314 human-annotated dialogues.
    - [RestBench](https://arxiv.org/abs/2306.06624): Simulates REST API applications with TMDB (54 tools) and Spotify (40 tools) scenarios.
    - [ToolHop](https://arxiv.org/abs/2501.02506): Tests multi-hop reasoning across 3,912 locally executable tools requiring 3-7 sequential calls.
- **Embodied Agent Benchmarks:**
    - [ALFWorld](https://arxiv.org/abs/2010.03768): Text-based embodied AI environment where agents complete household tasks using 9 basic actions.
- **Web Navigation Benchmarks:**
    - [WebShop](https://arxiv.org/abs/2207.01206): Online shopping simulation requiring agents to search and navigate products to fulfill user requirements.
- **Deep Research Benchmarks:** 
    - [GAIA](https://arxiv.org/abs/2311.12983): Complex information-seeking tasks requiring web search, browsing, VQA, code execution, and file processing.
    - [Humanity's Last Exam (HLE)](https://arxiv.org/abs/2501.14249): Extremely challenging reasoning problems testing advanced capabilities with code, search, and VQA tools. For efficient testing, we sampled 500 questions from the full set with 2,500 questions.

All the pre-processed data can be found in the `./data/` directory, except for ToolBench which needs to be downloaded from [ToolBench's official repository](https://github.com/OpenBMB/ToolBench), as it is too large to be included in our repository.

</details>

<details open>
<summary><h3>🤖 Model Serving</h3></summary>
Before running DeepAgent, ensure your reasoning model and auxiliary model are served using vLLM. DeepAgent is designed to work with powerful reasoning models as the main agent and can use an auxiliary model for tasks like memory generation and tool selection. For more details, please refer to [vLLM](https://github.com/vllm-project/vllm).

For the main reasoning model, we recommend using the following models. Performance improves from top to bottom, but computational cost also increases accordingly. You can choose a cost-effective model based on your needs:

| Model | Size | Type | Link |
|-------|------|------|---------|
| Qwen3-4B-Thinking | 4B | Thinking | [🤗 HuggingFace](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) |
| Qwen3-8B | 8B | Hybrid | [🤗 HuggingFace](https://huggingface.co/Qwen/Qwen3-8B) |
| Qwen3-30B-A3B-Thinking | 30B | Thinking | [🤗 HuggingFace](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) |
| QwQ-32B | 32B | Thinking | [🤗 HuggingFace](https://huggingface.co/Qwen/QwQ-32B) |
| Qwen3-235B-A22B-Thinking | 235B | Thinking | [🤗 HuggingFace](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507) |

For the auxiliary model, we recommend using the [Qwen2.5-Instruct](https://huggingface.co/collections/Qwen/qwen25) or [Qwen3-Instruct](https://huggingface.co/collections/Qwen/qwen3) series models with similar parameters to the main reasoning model, but without thinking capabilities for faster inference.

</details>

<details open>
<summary><h3>⚙️ Configuration</h3></summary>

All configurations are in `./config/base_config.yaml`, including API keys, service URLs and paths. You need to modify them to your actual configurations:

<details open>
<summary><h4>1. API Configuration</h4></summary>

Choose your task and configure the corresponding APIs:

- **ToolBench (RapidAPI):**
    - `toolbench_api`: RapidAPI key used in ToolBench. You can get it from [ToolBench's official repository](https://github.com/RapidAPI/ToolBench).
    - `toolbench_service_url`: ToolBench service URL. Keep it as default to use ToolBench's official service.
- **Deep Research:**
    - `google_serper_api`: Google Serper API key for web search. You can apply it [here](https://serper.dev/).
    - `use_jina`: Whether to use Jina Reader for stable URL content fetching.
    - `jina_api_key`: Jina API key. You can apply it [here](https://jina.ai/api-dashboard/reader).
- **RestBench (TMDB & Spotify):**
    - `tmdb_access_token`: TMDB access token. You can get the TMDB API key [here](https://developer.themoviedb.org/docs/getting-started).
    - `spotify_client_id`: Spotify client ID. You can get the Spotify API key [here](https://developer.spotify.com/documentation/web-api).
    - `spotify_client_secret`: Spotify client secret.
    - `spotify_redirect_uri`: Spotify redirect URI.
- **WebShop:**
    - `webshop_service_url`: WebShop service URL. You can create a new environment and serve it locally following the instructions in [WebShop's official repository](https://github.com/princeton-nlp/webshop).

</details>

<details open>
<summary><h4>2. Model Configuration</h4></summary>

Configure your model endpoints in the config file:

- **Main Reasoning LLM:**
    - `model_name`: The name of your served reasoning model (e.g., `QwQ-32B`).
    - `base_url`: API endpoint for your reasoning model service (e.g., `http://0.0.0.0:8080/v1`).
    - `api_key`: API key for accessing the reasoning model service. Set to `empty` if you are using vLLM.
    - `tokenizer_path`: Local path to the tokenizer files for the reasoning model.

- **Auxiliary LLM:**
    - `aux_model_name`: The name of your served auxiliary model (e.g., `Qwen2.5-32B-Instruct`).
    - `aux_base_url`: API endpoint for the auxiliary model service.
    - `aux_api_key`: API key for the auxiliary model. Set to `empty` if you are using vLLM.
    - `aux_tokenizer_path`: Local path to the tokenizer files for the auxiliary model.

- **VQA Model (for GAIA & HLE with image input):**
    - `vqa_model_name`: The name of your served vision-language model (e.g., `Qwen2.5-VL-32B-Instruct`). Model serving method is [here](#model-serving).
    - `vqa_base_url`: API endpoint for the VQA model service.
    - `vqa_api_key`: API key for the VQA model. Set to `empty` if you are using vLLM.

- **Tool Retriever:**
    - `tool_retriever_model_path`: Local path to the tool retriever model (e.g., `./models/bge-large-en-v1.5`).
    - `tool_retriever_api_base`: API endpoint for the tool retriever service. Pre-serving it can avoid reloading the retriever model every time you run the system. You can deploy it using the following command:
    ```bash
    python src/run_tool_search_server.py \
        --base_config_path ./config/base_config.yaml \
        --datasets toolbench,toolhop,tmdb,spotify,api_bank \
        --host 0.0.0.0 \
        --port 8001
    ```

</details>

<details open>
<summary><h4>3. Data Path Configuration</h4></summary>

All benchmark datasets are stored in the `./data/` directory. You can modify these paths if needed.

</details>

</details>

## 🚀 Run DeepAgent

To run on a benchmark dataset with tool search enabled, use the following command:
    
```bash
python src/run_deep_agent.py \
    --config_path ./config/base_config.yaml \
    --dataset_name toolbench \
    --enable_tool_search \
    --eval
```

To run on a benchmark dataset with closed-set mode, use the following command:

```bash
python src/run_deep_agent.py \
    --config_path ./config/base_config.yaml \
    --dataset_name gaia \
    --eval
```

**Parameters Explanation:**
- `--config_path`: Path to the main configuration file.
- `--dataset_name`: Name of the dataset to use (e.g., `toolbench`, `api_bank`, `tmdb`, `spotify`, `toolhop`, `gaia`, `hle`, `alfworld`, `webshop`).
- `--subset_num`: Number of samples to run from the dataset.
- `--concurrent_limit`: Maximum number of concurrent requests. Default is 32.
- `--enable_tool_search`: Allows the agent to search for tools. If disabled, it will only use the tools provided for the task (closed-set).
- `--enable_thought_folding`: Allows the agent to use the thought folding mechanism.
- `--max_action_limit`: Maximum number of actions (tool search and tool call) per question.
- `--max_fold_limit`: Maximum number of thought folds per question.
- `--top_k`: Maximum number of search tools to return.
- `--eval`: Run evaluation on the results after generation.



### Evaluation

Our model inference script can automatically save the model's input and output for evaluation. To run the evaluation, use the `--eval` flag when running `./src/run_deep_agent.py`. The evaluation scripts for each dataset are located in `./src/evaluate/`.



## 🔥 Deep Research Agent Family

<details open><summary>Welcome to try our deep research agent series: </summary><p>


> [**DeepAgent: A General Reasoning Agent with Scalable Toolsets (New!)**](https://arxiv.org/abs/2510.21618) <br>
> **TLDR:** An end-to-end deep reasoning agent that performs autonomous thinking, tool discovery, and action execution with brain-inspired memory folding mechanism. <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/RUC-NLPIR/DeepAgent) [![github](https://img.shields.io/github/stars/RUC-NLPIR/DeepAgent.svg?style=social)](https://github.com/RUC-NLPIR/DeepAgent) [![arXiv](https://img.shields.io/badge/Arxiv-2510.21618-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2510.21618) [![Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow?logo=huggingface)](https://huggingface.co/papers/2510.21618) [![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FXiaoxiLi0111%2Fstatus%2F1982649697467859438)](https://x.com/XiaoxiLi0111/status/1982649697467859438)

> [**Agentic Entropy-Balanced Policy Optimization**](https://arxiv.org/abs/2510.14545) <br>
> **TLDR:** An agentic RL algorithm designed to balance entropy in both the rollout and policy update phases. <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/RUC-NLPIR/ARPO) [![github](https://img.shields.io/github/stars/RUC-NLPIR/ARPO.svg?style=social)](https://github.com/RUC-NLPIR/ARPO) [![arXiv](https://img.shields.io/badge/Arxiv-2510.14545-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2510.14545) [![Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow?logo=huggingface)](https://huggingface.co/papers/2510.14545) [![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FKevin_GuoweiXu%2Fstatus%2F1858338565463421244)]()


> [**Agentic Reinforced Policy Optimization**](https://arxiv.org/abs/2507.19849) <br>
> **TLDR:** An agentic RL algorithm encourage the policy model to adaptively branch sampling during high-entropy tool-call rounds, <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/RUC-NLPIR/ARPO) [![github](https://img.shields.io/github/stars/RUC-NLPIR/ARPO.svg?style=social)](https://github.com/RUC-NLPIR/ARPO) [![arXiv](https://img.shields.io/badge/Arxiv-2507.19849-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.19849) [![Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow?logo=huggingface)](https://huggingface.co/papers/2507.19849) [![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FKevin_GuoweiXu%2Fstatus%2F1858338565463421244)](https://x.com/_akhaliq/status/1950172418250547478)

> [**Decoupled Planning and Execution: A Hierarchical Reasoning Framework for Deep Search**](https://arxiv.org/abs/2507.02652) <br>
> **TLDR:** This framework hierarchically decouples deep search into strategic planning and domain-specific execution by specialized agents. <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/RUC-NLPIR/HiRA) [![github](https://img.shields.io/github/stars/RUC-NLPIR/HiRA.svg?style=social)](https://github.com/RUC-NLPIR/HiRA) [![arXiv](https://img.shields.io/badge/Arxiv-2507.02652-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.02652) [![Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow?logo=huggingface)](https://huggingface.co/papers/2507.02652) [![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2Fdongxi_nlp%2Fstatus%2F1941223631033389301)](https://x.com/dongxi_nlp/status/1941223631033389301)


> [**Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning**](https://arxiv.org/abs/2505.16410) <br>
> **TLDR:** An end-to-end TIR post-training framework that empowers LLMs to autonomously interact with multi-tool environments through Self-Critic RL design<br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/RUC-NLPIR/Tool-Star) [![github](https://img.shields.io/github/stars/RUC-NLPIR/Tool-Star.svg?style=social)](https://github.com/RUC-NLPIR/Tool-Star) [![arXiv](https://img.shields.io/badge/Arxiv-2505.16410-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2505.16410) [![Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow?logo=huggingface)](https://huggingface.co/papers/2505.16410) [![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FKevin_GuoweiXu%2Fstatus%2F1858338565463421244)](https://x.com/_akhaliq/status/1925924431676821698)

 > [**WebThinker: Empowering Large Reasoning Models with Deep Research Capability (NeurIPS 2025)**](https://arxiv.org/abs/2504.21776) <br>
> **TLDR:** A deep research agent that empowers large reasoning models with autonomous search, web browsing, and research report drafting capabilities. <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/RUC-NLPIR/WebThinker) [![github](https://img.shields.io/github/stars/RUC-NLPIR/WebThinker.svg?style=social)](https://github.com/RUC-NLPIR/WebThinker) [![arXiv](https://img.shields.io/badge/Arxiv-2504.21776-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2504.21776) [![Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow?logo=huggingface)](https://huggingface.co/papers/2504.21776) [![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2Fkakakbibibi%2Fstatus%2F1917768235069628823)](https://x.com/kakakbibibi/status/1917768235069628823)

> [**Search-o1: Agentic Search-Enhanced Large Reasoning Models (EMNLP 2025)**](https://arxiv.org/abs/2501.05366) <br>
> **TLDR:** An agentic search-enhanced framework that integrates autonomous knowledge retrieval with large reasoning models through Agentic RAG and reasoning-in-documents modules. <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/RUC-NLPIR/Search-o1) [![github](https://img.shields.io/github/stars/RUC-NLPIR/Search-o1.svg?style=social)](https://github.com/RUC-NLPIR/Search-o1) [![arXiv](https://img.shields.io/badge/Arxiv-2501.16399-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2501.05366) [![Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow?logo=huggingface)](https://huggingface.co/papers/2501.05366) [![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2F_akhaliq%2Fstatus%2F1877584951840764166%3Ft%3DfnbTblnqhiPtAyYr1PHbbw%26s%3D19)](https://x.com/_akhaliq/status/1877584951840764166?t=fnbTblnqhiPtAyYr1PHbbw&s=19) 

</details>


## 📄 Citation

If you find this work helpful, please cite our paper:
```bibtex
@misc{deepagent,
      title={DeepAgent: A General Reasoning Agent with Scalable Toolsets}, 
      author={Xiaoxi Li and Wenxiang Jiao and Jiarui Jin and Guanting Dong and Jiajie Jin and Yinuo Wang and Hao Wang and Yutao Zhu and Ji-Rong Wen and Yuan Lu and Zhicheng Dou},
      year={2025},
      eprint={2510.21618},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.21618}, 
}
```

## 📄 License

This project is released under the [MIT License](LICENSE).

## 📞 Contact

For any questions or feedback, please reach out to us at [xiaoxi_li@ruc.edu.cn](xiaoxi_li@ruc.edu.cn).
