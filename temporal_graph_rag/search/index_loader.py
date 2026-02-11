"""Загрузчик индекса из Parquet + FAISS."""

import json
import logging
from pathlib import Path
from typing import Any

from temporal_graph_rag.models.data_models import (
    Community,
    Entity,
    Relationship,
    TemporalPosition,
    TextUnit,
)
from temporal_graph_rag.storage.parquet_storage import ParquetStorage
from temporal_graph_rag.storage.vector_storage import VectorStorage

logger = logging.getLogger(__name__)


class IndexLoader:
    """Загружает индекс из Parquet + FAISS."""

    def __init__(self, index_dir: str | Path) -> None:
        """
        Инициализация загрузчика.
        
        Args:
            index_dir: Директория с файлами индекса
        """
        self.index_path = Path(index_dir)
        if not self.index_path.exists():
            raise FileNotFoundError(f"Директория индекса не найдена: {self.index_path}")
        
        self.parquet_storage = ParquetStorage(self.index_path)
        self.vector_storage = VectorStorage(self.index_path)
        
    def load_index(
        self,
    ) -> tuple[list[TextUnit], dict[str, Entity], dict[str, Relationship], list[Community]]:
        """
        Загружает полный индекс из директории.

        Returns:
            Кортеж (text_units, entities_dict, relationships_dict, communities)
        """
        logger.info(f"Загрузка индекса из: {self.index_path}")

        # Загружаем табличные данные из Parquet
        text_units = self.parquet_storage.load_text_units()
        entities_dict = self.parquet_storage.load_entities()
        relationships_dict = self.parquet_storage.load_relationships()
        communities = self.parquet_storage.load_communities()
        
        # Загружаем FAISS индексы
        self.vector_storage.load_entity_index()
        self.vector_storage.load_community_index()
        self.vector_storage.load_text_unit_index()  # Опционально
        
        # Присоединяем эмбеддинги к text_units (если есть)
        for unit in text_units:
            embedding = self.vector_storage.get_text_unit_embedding(unit.id)
            if embedding:
                unit.embedding = embedding
        
        # Присоединяем эмбеддинги к entities
        for name in entities_dict:
            embedding = self.vector_storage.get_entity_embedding(name)
            if embedding:
                entities_dict[name].embedding = embedding
        
        # Присоединяем эмбеддинги к communities
        for comm in communities:
            embedding = self.vector_storage.get_community_embedding(comm.id)
            if embedding:
                comm.embedding = embedding
        
        logger.info("Индекс успешно загружен")
        return text_units, entities_dict, relationships_dict, communities

    def search_entities(self, query_embedding: list[float], k: int = 10) -> list[tuple[str, float]]:
        """
        Поиск наиболее похожих entities.
        
        Args:
            query_embedding: Вектор запроса
            k: Количество результатов
        
        Returns:
            Список кортежей (entity_name, similarity_score)
        """
        return self.vector_storage.search_entities(query_embedding, k)
    
    def search_communities(self, query_embedding: list[float], k: int = 5) -> list[tuple[str, float]]:
        """
        Поиск наиболее похожих communities.
        
        Args:
            query_embedding: Вектор запроса
            k: Количество результатов
        
        Returns:
            Список кортежей (community_id, similarity_score)
        """
        return self.vector_storage.search_communities(query_embedding, k)
    
    def search_text_units(self, query_embedding: list[float], k: int = 10) -> list[tuple[str, float]]:
        """
        Поиск наиболее похожих text_units (для классического RAG).
        
        Args:
            query_embedding: Вектор запроса
            k: Количество результатов
        
        Returns:
            Список кортежей (text_unit_id, similarity_score)
        """
        return self.vector_storage.search_text_units(query_embedding, k)

