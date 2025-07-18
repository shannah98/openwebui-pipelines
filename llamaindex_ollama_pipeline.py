"""
title: Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from the OpenWebUI knowledge base using Llama Index and ChromaDB.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama, llama-index-vector-stores-chroma, chromadb
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import os

from pydantic import BaseModel


class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

    def __init__(self):
        self.documents = None
        self.index = None

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
            }
        )

    async def on_startup(self):
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.core import Settings, VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore
        import chromadb
 
        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        # This function is called when the server is started. It connects to the
        # existing ChromaDB vector store used by OpenWebUI.
       # db_path = "/app/backend/data/chroma"
       #his function is called when the server is started. It connects to the directory of uploaded files
        db_path = "/app/backend/data/uploads"
 
        if not os.path.exists(db_path):
            print(
                f"ChromaDB path not found at {db_path}. Please ensure you have added documents through the OpenWebUI interface."
            )
            self.index = None
            return
 
        db = chromadb.PersistentClient(path=db_path)
        # The default collection name in OpenWebUI is 'webui'
        chroma_collection = db.get_or_create_collection("webui")
 
        if chroma_collection.count() == 0:
            print("Knowledge base is empty. Please add documents through the OpenWebUI interface.")
            self.index = None
            return
 
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.index = VectorStoreIndex.from_vector_store(vector_store)

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)

        if self.index is None:
            return "Knowledge base not initialized or is empty. Please add documents through the OpenWebUI interface."
 
        query_engine = self.index.as_query_engine(streaming=True, llm=Settings.llm)
        response = query_engine.query(user_message)
 
        return response.response_gen
