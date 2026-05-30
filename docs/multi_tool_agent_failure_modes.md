# Multi tool agent failure modes and troubleshooting

This page is a small troubleshooting note for DeepAgent users who run large multi tool tasks such as ToolBench, API-Bank, ToolHop, GAIA and HLE.  
It does not change any code. The goal is to give a repeatable checklist when the agent looks stuck or behaves strangely around tools.

Use this page when you see things like:

- obvious wrong tools selected although correct tools exist
- the same tool being called again and again with no real progress
- correct tool results, but the final answer ignores them
- long chains of failed tool calls after a configuration change

The sections below keep the scope narrow and close to DeepAgent as released in this repository.

---

## 1. Quick checklist (one screen)

When something looks wrong, run through this short list first.

1. **Confirm the scenario**

   - Which dataset or task are you running  
     (for example `toolbench`, `api_bank`, `toolhop`, `gaia`, `hle`, `alfworld`, `webshop`)  
   - Are you using open tool search or a closed tool set  
     (check whether `--enable_tool_search` is on)

2. **Check configuration basics**

   - `config/base_config.yaml` is the file actually passed to `--config_path`
   - API keys and service URLs are set for the dataset you are using
   - `tool_retriever_model_path` or tool retriever endpoint, if tool search is enabled

3. **Look at the trace**

   - save or inspect the model thought steps and tool calls
   - note which tools are selected, with which arguments
   - note whether tool results are read, summarised, or ignored

4. **Match symptoms to the patterns below**

   - wrong tool chosen
   - right tool but wrong arguments
   - good tool result, bad final answer
   - hidden environment or config mismatch
   - exhaustion of action budget without progress

5. **Apply the minimal fix**

   - change one thing at a time
   - keep a small log of what changed and the new behaviour
   - once a fix works, keep that configuration as a known good baseline

---

## 2. Typical failure modes in multi tool settings

This section describes the most common patterns we have seen in multi tool agents, together with minimal checks that fit DeepAgent.

Each pattern is written as:

- symptom  
- likely cause  
- minimal checks in DeepAgent

### 2.1 Wrong tool selected despite good descriptions

**Symptom**

The agent repeatedly chooses tools that are clearly unrelated to the user goal, even though a better tool exists in the same tool set.

**Likely causes**

- tool descriptions are too similar or too long  
- the retriever index was built with different descriptions than the ones shown to the model  
- the dataset configuration points to a different tool set than the one you expect

**Minimal checks in DeepAgent**

- Open `config/base_config.yaml` and confirm the dataset you run (for example `toolbench`, `api_bank`, `toolhop`) matches the one used when building the tool index.  
- Check the tool retriever configuration:

  - `tool_retriever_model_path` or retriever service URL  
  - any custom index files or pre processed tool metadata

- Inspect a few failing cases and compare the natural language description of the wrong tool and the tool you think should be chosen. If they share most keywords, consider simplifying or sharpening the descriptions before training or indexing.

### 2.2 Right tool chosen but arguments are wrong or missing

**Symptom**

The agent selects the correct tool, but calls it with missing fields, wrong types, or obviously incorrect values.  
You may see many HTTP `4xx` errors, schema validation errors, or empty results.

**Likely causes**

- function schema does not fully describe argument constraints  
- example calls in the prompt do not match the real schema  
- dataset specific wrappers around tools perform extra validation that the model never sees

**Minimal checks in DeepAgent**

- Verify the tool schema or OpenAPI spec used to construct the tool. For ToolBench like settings, confirm that the schema passed to the model is the same as the one used by the backend.  
- Check whether there are intermediate wrappers in `src/` that convert between model friendly arguments and actual API calls. Make sure the names and types match.  
- For a few failing runs, log the exact arguments that DeepAgent sends to the tool. Compare them with a manual call that succeeds.

### 2.3 Tool result is correct but the final answer ignores it

**Symptom**

Tools appear to run successfully, but the final natural language answer contradicts or ignores the tool outputs.  
For example, the agent fetches data from an API, then answers with a guess that does not use that data.

**Likely causes**

- the prompt does not strongly require the model to ground its final answer in tool results  
- long context causes the tool result to be pushed far away from the current focus of attention  
- additional reasoning turns overwrite or compress the tool output before the final answer

**Minimal checks in DeepAgent**

- Inspect the thought trace. Confirm that tool outputs are present in the context when the final answer is generated.  
- Strengthen any system messages or templates that say how DeepAgent should use tools, for example by explicitly asking the model to quote or summarise tool outputs.  
- If the context is very long, consider reducing irrelevant history or limiting the number of previous actions kept in the active window.

### 2.4 Hidden configuration or environment mismatch

**Symptom**

The same script works on one machine but not another. Tools time out, return unexpected errors, or behave differently across runs.

**Likely causes**

- environment variables differ between shells or machines  
- API keys refer to different accounts with different rate limits or permissions  
- network settings block some endpoints

**Minimal checks in DeepAgent**

- Confirm that the `base_config.yaml` file used in your command is the one you edited. Avoid having multiple similar config files in different folders.  
- Print or log the effective configuration at startup, especially API keys (masked), base URLs, and dataset names.  
- Test a small manual call to a failing tool outside DeepAgent, using the same key and endpoint, to see whether the problem is in the agent or in the external service.

### 2.5 Action budget exhausted with no real progress

**Symptom**

The agent burns through many actions repeating similar tool calls, or oscillates between a small set of tools without converging to a solution.

**Likely causes**

- `max_action_limit` or related parameters are high, but the underlying plan never reaches a terminal condition  
- tool results are noisy or partial, so the model keeps trying to repair the situation  
- reward, success, or stopping conditions are under specified for the benchmark

**Minimal checks in DeepAgent**

- Check the run command for values of `--max_action_limit`, `--max_fold_limit`, and related flags.  
- Inspect a trace where the agent hits the limit. Identify the earliest point where behaviour started to loop or stall.  
- If possible, add clearer intermediate goals or success checks in the prompt, so that the agent can terminate earlier once enough information has been collected.

---

## 3. When filing an issue with DeepAgent

If you want to report a problem to the DeepAgent repository, including the following information usually makes triage much faster:

- exact command line, including `--dataset_name`, `--config_path`, and tool search flags  
- a short description of what you expected and what actually happened  
- one or two anonymised traces:

  - tool name  
  - arguments  
  - tool output (trimmed)  
  - final answer

This lets maintainers decide whether the issue is about configuration, model behaviour, or a real bug in the framework.

---

## 4. References

This checklist is based on practical experience with multi tool agents and common failure modes in LLM tool pipelines.  
Some of the patterns were inspired by open source failure mode maps such as the WFGY Problem Map for LLM, RAG and agent systems.
