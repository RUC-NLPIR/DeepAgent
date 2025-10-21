import time
import os
import hashlib
import torch
from sentence_transformers import SentenceTransformer, util


class ToolRetriever:
    def __init__(self, corpus, corpus2tool, model_path, cache_dir, load_cache=True, corpus_identifier=""):
        self.corpus = corpus
        self.corpus2tool = corpus2tool
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.load_cache = load_cache
        self.corpus_identifier = corpus_identifier if corpus_identifier else self._generate_corpus_identifier()

        self.embedder = self.build_retrieval_embedder()
        self.corpus_embeddings = self.build_corpus_embeddings()

    def _generate_corpus_identifier(self):
        corpus_str = "".join(self.corpus)
        return hashlib.md5(corpus_str.encode('utf-8')).hexdigest()

    def build_retrieval_embedder(self):
        print("Loading embedding model...")
        embedder = SentenceTransformer(self.model_path)
        return embedder

    def get_cache_path(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        unique_str = self.model_path + "_" + self.corpus_identifier
        cache_name = hashlib.md5(unique_str.encode('utf-8')).hexdigest() + ".pt"
        return os.path.join(self.cache_dir, cache_name)

    def build_corpus_embeddings(self):
        print("Building corpus embeddings...")
        cache_path = self.get_cache_path()
        # if os.path.exists(cache_path) and self.load_cache:
        #     print(f"Loading corpus embeddings from cache: {cache_path}")
        #     corpus_embeddings = torch.load(cache_path)
        #     return corpus_embeddings

        formatted_corpus = []
        for text in self.corpus:
            if "e5" in self.model_path.lower():
                formatted_text = f"passage: {text}"
            elif "bge" in self.model_path.lower():
                formatted_text = text
            else:
                formatted_text = text
            formatted_corpus.append(formatted_text)

        start = time.time()
        if "bge" in self.model_path.lower():
            corpus_embeddings = self.embedder.encode(formatted_corpus, normalize_embeddings=True, convert_to_tensor=True)
        else:
            corpus_embeddings = self.embedder.encode(formatted_corpus, convert_to_tensor=True)
        print(f"Corpus embeddings calculated in {time.time() - start:.2f} seconds.")

        torch.save(corpus_embeddings, cache_path)
        print(f"Corpus embeddings saved to cache: {cache_path}")
        return corpus_embeddings

    def retrieving(self, query, top_k=10):
        # print("Retrieving...")
        start = time.time()
        if "e5" in self.model_path.lower():
            formatted_query = f"query: {query}"
        elif "bge" in self.model_path.lower():
            formatted_query = query
        else:
            formatted_query = query
        
        query_embedding = self.embedder.encode(
            formatted_query,
            normalize_embeddings=("bge" in self.model_path.lower()),
            convert_to_tensor=True
        )

        # print("Calculating hits...")
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k, score_function=util.cos_sim)
        # print(f"Hits calculated in {time.time() - start:.2f} seconds.")
        
        retrieved_tools = []
        for hit in hits[0]:
            tool_doc = self.corpus2tool[self.corpus[hit['corpus_id']]]
            retrieved_tools.append(tool_doc)
            
        return retrieved_tools

