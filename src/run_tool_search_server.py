import argparse
import json
import os
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tools.rapid_api import RapidAPIRetriever
from tools.toolhop import ToolHopRetriever
from tools.restbench_api import RestBenchRetriever
from tools.api_bank import APIBankRetriever
import yaml
import asyncio

class RetrieveRequest(BaseModel):
    dataset_name: str
    query: str
    top_k: int = 10
    executable_tools: Optional[List[Dict[str, Any]]] = None  # ToolHop narrowing

class RetrieveResponse(BaseModel):
    results: List[Dict[str, Any]]

def build_retriever(dataset_name: str, cfg: Dict[str, Any]):
    cache_dir = cfg.get('tool_index_cache_dir', './cache/tool_index')
    model_path = cfg.get('tool_retriever_model_path', '')
    if dataset_name == 'toolbench':
        corpus_tsv_path = cfg.get('toolbench_corpus_tsv_path', '')
        return RapidAPIRetriever(
            corpus_tsv_path=corpus_tsv_path,
            model_path=model_path,
            cache_dir=cache_dir,
            load_cache=False,
        )
    if dataset_name == 'toolhop':
        corpus_json_path = cfg.get('toolhop_data_path', '')
        return ToolHopRetriever(
            corpus_json_path=corpus_json_path,
            model_path=model_path,
            cache_dir=cache_dir,
            load_cache=False,
        )
    if dataset_name in ['tmdb', 'spotify']:
        return RestBenchRetriever(
            dataset_name=dataset_name,
            model_path=model_path,
            cache_dir=cache_dir,
            args=type("Args", (), cfg),
            load_cache=False,
        )
    # if dataset_name == 'api_bank':
    #     apis_dir = cfg.get('api_bank_lv3_apis_dir', '')
    #     return APIBankRetriever(
    #         model_path=model_path,
    #         apis_dir=apis_dir,
    #         cache_dir=cache_dir,
    #         load_cache=False,
    #     )
    return None

def _prewarm(retriever, dataset_name: str):
    try:
        if dataset_name == 'toolhop':
            # ToolHop可选executables限制；预热用空列表即可
            retriever.retrieving("warmup", 1, [])
        else:
            retriever.retrieving("warmup", 1)
    except Exception:
        pass

def create_app(cfg: Dict[str, Any], preload_datasets: List[str]) -> FastAPI:
    app = FastAPI(title="Tool Retrieval Server")

    # 1) 预构建并缓存所有检索器
    retriever_cache: Dict[str, Any] = {}
    for ds in preload_datasets:
        retr = build_retriever(ds, cfg)
        if retr is not None:
            retriever_cache[ds] = retr
            # 2) 预热一次，触发模型加载与索引构建
            _prewarm(retr, ds)

    # 检索超时（秒），优先用 cfg，其次环境变量，默认 25s
    try:
        timeout_s = int(cfg.get('tool_retrieval_timeout', os.getenv('TOOL_RETRIEVAL_TIMEOUT', 25)))
    except Exception:
        timeout_s = 25

    @app.post("/retrieve", response_model=RetrieveResponse)
    async def retrieve(req: RetrieveRequest):
        try:
            ds = req.dataset_name
            if ds not in retriever_cache:
                # 未配置的数据集直接报错，避免在线构建
                raise HTTPException(status_code=400, detail=f"Unsupported or not preloaded dataset: {ds}")

            retriever = retriever_cache[ds]

            if ds == 'toolhop' and req.executable_tools is not None:
                results = await asyncio.wait_for(
                    asyncio.to_thread(retriever.retrieving, req.query, req.top_k, req.executable_tools),
                    timeout=timeout_s
                )
            else:
                results = await asyncio.wait_for(
                    asyncio.to_thread(retriever.retrieving, req.query, req.top_k),
                    timeout=timeout_s
                )

            if not isinstance(results, list):
                results = []
            return RetrieveResponse(results=results)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail=f"Retrieval timeout after {timeout_s}s for dataset: {req.dataset_name}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "preloaded": list(retriever_cache.keys())}

    return app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config_path", default="./config/base_config.yaml", type=str)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    # 可选：仅加载指定数据集，默认全部加载
    parser.add_argument("--datasets", type=str, default="toolbench,toolhop,tmdb,spotify,api_bank",
                        help="Comma-separated datasets to preload: toolbench,toolhop,tmdb,spotify,api_bank")
    args = parser.parse_args()

    with open(args.base_config_path, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f) or {}

    # 先拷贝 base_config.yaml 的全部键，保证 RestBench 所需的 tmdb/spotify 凭证存在
    retr_cfg = dict(base_cfg)

    # 用你的绝对路径覆盖关键索引/语料路径（其余保持 base_cfg）
    retr_cfg.update({
        'tool_retriever_model_path': './LLMs/Retrievers/bge-large-en-v1.5',
        'tool_index_cache_dir': './cache/tool_index',
        'toolbench_corpus_tsv_path': './data/ToolBench/retrieval/G1/corpus.tsv',
        'toolhop_data_path': './data/ToolHop/ToolHop.json',
        'api_bank_apis_dir': './data/API-Bank/apis',
        'api_bank_lv3_apis_dir': './data/API-Bank/lv3_apis_abs_path',
        'tmdb_toolset_path': './data/RestBench/specs/tmdb_oas.json',
        'spotify_toolset_path': './data/RestBench/specs/spotify_oas.json',
    })

    # 可选：确保缓存目录存在
    os.makedirs(retr_cfg.get('tool_index_cache_dir', './cache/tool_index'), exist_ok=True)

    preload_datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    app = create_app(retr_cfg, preload_datasets)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main() 