import logging
import os
from pathlib import Path

from llama_index.core import PropertyGraphIndex
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core import StorageContext
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

logger = logging.getLogger(__name__)

GRAPH_DIR = Path("qdrant_data/graph_store")

class GraphStoreManager:
    def __init__(self, model_name: str = "llama-3.1-8b-instant", embed_model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.embed_model_name = embed_model_name
        self.llm = None
        self.embed_model = None

        GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        self.persist_dir = str(GRAPH_DIR)
        self.index = None

    def _ensure_clients(self):
        if self.llm is None:
            self.llm = Groq(model=self.model_name, api_key=os.getenv("GROQ_API_KEY", ""))
        if self.embed_model is None:
            self.embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name)

    def load_or_create(self, documents: list[Document] = None):
        if os.path.exists(os.path.join(self.persist_dir, "graph_store.json")):
            self._ensure_clients()
            logger.info("Loading existing PropertyGraphIndex...")
            graph_store = SimplePropertyGraphStore.from_persist_dir(self.persist_dir)
            storage_context = StorageContext.from_defaults(property_graph_store=graph_store)
            self.index = PropertyGraphIndex.from_existing(
                property_graph_store=graph_store,
                llm=self.llm,
                embed_model=self.embed_model
            )
        elif documents:
            self._ensure_clients()
            logger.info("Creating new PropertyGraphIndex from documents...")
            graph_store = SimplePropertyGraphStore()
            storage_context = StorageContext.from_defaults(property_graph_store=graph_store)
            self.index = PropertyGraphIndex.from_documents(
                documents,
                storage_context=storage_context,
                llm=self.llm,
                embed_model=self.embed_model,
                show_progress=True
            )
            self.index.property_graph_store.persist(self.persist_dir)
        else:
            logger.warning("No graph store found and no documents provided.")

        return self.index

    def search_synergies(self, query: str, top_k: int = 3):
        if not self.index:
            return []
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        return [n.get_text() for n in nodes]
