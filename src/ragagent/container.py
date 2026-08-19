from __future__ import annotations

from ragagent.agent import AgentService, EvidenceGate
from ragagent.config import Settings
from ragagent.db import Database
from ragagent.dense_index import DenseIndexManager
from ragagent.ingestion import IngestionService, JobProcessor, ObjectStore
from ragagent.models import build_chat_client, build_embedding_client, build_rerank_client
from ragagent.repositories import Repository
from ragagent.retrieval import HybridRetriever


class Container:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.database = Database(settings.database_path)
        self.repository = Repository(self.database)
        self.dense_index = DenseIndexManager(self.repository, settings)
        self.object_store = ObjectStore(settings)
        self.embedding_client = build_embedding_client(settings)
        self.rerank_client = build_rerank_client(settings)
        self.chat_client = build_chat_client(settings)
        self.ingestion = IngestionService(self.repository, self.object_store, settings)
        self.retriever = HybridRetriever(
            self.repository,
            self.embedding_client,
            self.rerank_client,
            settings,
            self.dense_index,
        )
        self.evidence_gate = EvidenceGate(settings, self.chat_client)
        self.agent = AgentService(
            self.repository,
            self.retriever,
            self.evidence_gate,
            self.chat_client,
            settings,
        )
        self.job_processor = JobProcessor(
            self.repository,
            self.ingestion,
            self.embedding_client,
            self.dense_index,
        )
        self.job_processor.agent_service = self.agent

    def initialize(self) -> None:
        self.database.initialize()
        self.object_store.initialize()
        self.dense_index.initialize()
