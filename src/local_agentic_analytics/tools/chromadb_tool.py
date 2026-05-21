"""Persistent ChromaDB helper with local sentence-transformers embeddings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from local_agentic_analytics.core.config import PROJECT_ROOT, load_config


DEFAULT_PERSIST_DIRECTORY = PROJECT_ROOT / "databases" / "chromadb"
DEFAULT_COLLECTION_NAME = "local_agentic_analytics"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbeddingFunction:
    """Lazy sentence-transformers embedding function for ChromaDB."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        if not model_name or not model_name.strip():
            raise ValueError("model_name must not be empty")

        self.model_name = model_name.strip()
        self._model = None

    def __call__(self, input: list[str]) -> list[list[float]]:
        model = self._load_model()
        embeddings = model.encode(
            list(input),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

        return self._model


class ChromaDBTool:
    """Small persistent ChromaDB wrapper for local RAG experiments."""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedding_function: Any | None = None,
        anonymized_telemetry: bool = False,
    ):
        if not collection_name or not collection_name.strip():
            raise ValueError("collection_name must not be empty")

        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name.strip()
        self.embedding_model_name = embedding_model_name
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_function = embedding_function or SentenceTransformerEmbeddingFunction(
            embedding_model_name
        )
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=anonymized_telemetry),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    @classmethod
    def from_config(cls, config_path: str = "chromadb.yaml") -> "ChromaDBTool":
        config = load_config(config_path)
        chroma_config = config.get("chromadb", {})
        embedding_config = config.get("embedding", {})

        if not isinstance(chroma_config, dict):
            raise ValueError("chromadb config must be a mapping")
        if not isinstance(embedding_config, dict):
            raise ValueError("embedding config must be a mapping")

        provider = str(embedding_config.get("provider", "sentence-transformers"))
        if provider != "sentence-transformers":
            raise ValueError(f"Unsupported ChromaDB embedding provider: {provider}")

        persist_directory = _resolve_project_path(
            str(chroma_config.get("persist_directory", DEFAULT_PERSIST_DIRECTORY))
        )
        collection_name = str(
            chroma_config.get("collection_name", DEFAULT_COLLECTION_NAME)
        )
        model_name = str(embedding_config.get("model", DEFAULT_EMBEDDING_MODEL))
        anonymized_telemetry = bool(chroma_config.get("anonymized_telemetry", False))

        return cls(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_model_name=model_name,
            anonymized_telemetry=anonymized_telemetry,
        )

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return int(self.collection.count())

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict],
    ) -> dict[str, Any]:
        """Add documents to the collection, skipping deterministic duplicates."""
        self._validate_documents(documents, metadatas)

        ids = [
            _build_document_id(document=document, metadata=metadata)
            for document, metadata in zip(documents, metadatas)
        ]
        existing_ids = self._get_existing_ids(ids)

        new_ids: list[str] = []
        new_documents: list[str] = []
        new_metadatas: list[dict] = []
        seen_new_ids: set[str] = set()
        for document_id, document, metadata in zip(ids, documents, metadatas):
            if document_id in existing_ids or document_id in seen_new_ids:
                continue

            seen_new_ids.add(document_id)
            new_ids.append(document_id)
            new_documents.append(document)
            new_metadatas.append(metadata)

        if new_ids:
            self.collection.add(
                ids=new_ids,
                documents=new_documents,
                metadatas=new_metadatas,
            )

        return {
            "added": len(new_ids),
            "skipped": len(ids) - len(new_ids),
            "ids": new_ids,
        }

    def query(self, text: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Query the collection and return compact ranked matches."""
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be greater than 0")
        if self.count() == 0:
            return []

        result = self.collection.query(
            query_texts=[text.strip()],
            n_results=top_k,
        )
        return _flatten_query_result(result)

    @staticmethod
    def _validate_documents(documents: list[str], metadatas: list[dict]) -> None:
        if not documents:
            raise ValueError("documents must not be empty")
        if len(documents) != len(metadatas):
            raise ValueError("documents and metadatas must have the same length")
        if any(not isinstance(document, str) or not document.strip() for document in documents):
            raise ValueError("all documents must be non-empty strings")
        if any(not isinstance(metadata, dict) for metadata in metadatas):
            raise ValueError("all metadatas must be dictionaries")

    def _get_existing_ids(self, ids: list[str]) -> set[str]:
        existing = self.collection.get(ids=ids)
        return set(existing.get("ids", []))


def _flatten_query_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    matches = []
    for index, document_id in enumerate(ids):
        matches.append(
            {
                "id": document_id,
                "document": documents[index],
                "metadata": metadatas[index],
                "distance": distances[index],
            }
        )

    return matches


def _build_document_id(document: str, metadata: dict) -> str:
    payload = {
        "document": document.strip(),
        "metadata": metadata,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_project_path(path: str) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved

    return PROJECT_ROOT / resolved
