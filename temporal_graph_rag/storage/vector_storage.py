"""Модуль для работы с FAISS векторным хранилищем."""

import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VectorStorage:
    """
    Класс для работы с FAISS векторными индексами.
    
    Структура:
    - vectors/entities.faiss: FAISS индекс для entities
    - vectors/entities_mapping.parquet: маппинг faiss_id -> entity_name
    - vectors/communities.faiss: FAISS индекс для communities
    - vectors/communities_mapping.parquet: маппинг faiss_id -> community_id
    """

    def __init__(self, base_path: str | Path) -> None:
        """
        Инициализация векторного хранилища.
        
        Args:
            base_path: Базовый путь для хранения данных
        """
        self.base_path = Path(base_path)
        self.vectors_path = self.base_path / "vectors"
        self.vectors_path.mkdir(parents=True, exist_ok=True)
        
        # Кэш для загруженных индексов
        self._entities_index: faiss.Index | None = None
        self._communities_index: faiss.Index | None = None
        self._text_units_index: faiss.Index | None = None
        self._entities_mapping: pd.DataFrame | None = None
        self._communities_mapping: pd.DataFrame | None = None
        self._text_units_mapping: pd.DataFrame | None = None

    def save_entity_vectors(
        self,
        entity_names: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """
        Сохраняет векторы entities в FAISS индекс.
        
        Args:
            entity_names: Список имен сущностей (в том же порядке, что и embeddings)
            embeddings: Список эмбеддингов
        """
        logger.info(f"Сохранение {len(embeddings)} entity векторов в FAISS...")
        
        # Конвертируем в numpy array
        vectors = np.array(embeddings, dtype=np.float32)
        dimension = vectors.shape[1]
        
        # Создаем FAISS индекс (L2 distance)
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        
        # Сохраняем индекс
        index_file = self.vectors_path / "entities.faiss"
        faiss.write_index(index, str(index_file))
        logger.info(f"Entities FAISS индекс сохранен: {index_file}")
        
        # Создаем и сохраняем маппинг
        mapping_df = pd.DataFrame({
            "faiss_id": range(len(entity_names)),
            "entity_name": entity_names,
        })
        mapping_file = self.vectors_path / "entities_mapping.parquet"
        mapping_df.to_parquet(mapping_file, index=False)
        logger.info(f"Entities mapping сохранен: {mapping_file}")

    def save_community_vectors(
        self,
        community_ids: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """
        Сохраняет векторы communities в FAISS индекс.
        
        Args:
            community_ids: Список ID communities (в том же порядке, что и embeddings)
            embeddings: Список эмбеддингов
        """
        logger.info(f"Сохранение {len(embeddings)} community векторов в FAISS...")
        
        # Конвертируем в numpy array
        vectors = np.array(embeddings, dtype=np.float32)
        dimension = vectors.shape[1]
        
        # Создаем FAISS индекс (L2 distance)
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        
        # Сохраняем индекс
        index_file = self.vectors_path / "communities.faiss"
        faiss.write_index(index, str(index_file))
        logger.info(f"Communities FAISS индекс сохранен: {index_file}")
        
        # Создаем и сохраняем маппинг
        mapping_df = pd.DataFrame({
            "faiss_id": range(len(community_ids)),
            "community_id": community_ids,
        })
        mapping_file = self.vectors_path / "communities_mapping.parquet"
        mapping_df.to_parquet(mapping_file, index=False)
        logger.info(f"Communities mapping сохранен: {mapping_file}")

    def save_text_unit_vectors(
        self,
        text_unit_ids: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """
        Сохраняет векторы text_units в FAISS индекс (для сравнения с классическим RAG).
        
        Args:
            text_unit_ids: Список ID text_units (в том же порядке, что и embeddings)
            embeddings: Список эмбеддингов
        """
        logger.info(f"Сохранение {len(embeddings)} text_unit векторов в FAISS...")
        
        # Конвертируем в numpy array
        vectors = np.array(embeddings, dtype=np.float32)
        dimension = vectors.shape[1]
        
        # Создаем FAISS индекс (L2 distance)
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        
        # Сохраняем индекс
        index_file = self.vectors_path / "text_units.faiss"
        faiss.write_index(index, str(index_file))
        logger.info(f"Text units FAISS индекс сохранен: {index_file}")
        
        # Создаем и сохраняем маппинг
        mapping_df = pd.DataFrame({
            "faiss_id": range(len(text_unit_ids)),
            "text_unit_id": text_unit_ids,
        })
        mapping_file = self.vectors_path / "text_units_mapping.parquet"
        mapping_df.to_parquet(mapping_file, index=False)
        logger.info(f"Text units mapping сохранен: {mapping_file}")

    def load_entity_index(self) -> None:
        """Загружает entities FAISS индекс и маппинг в память."""
        if self._entities_index is not None:
            return  # Уже загружен
        
        index_file = self.vectors_path / "entities.faiss"
        mapping_file = self.vectors_path / "entities_mapping.parquet"
        
        logger.info(f"Загрузка entities FAISS индекса из {index_file}...")
        self._entities_index = faiss.read_index(str(index_file))
        self._entities_mapping = pd.read_parquet(mapping_file)
        logger.info(
            f"Entities индекс загружен: {self._entities_index.ntotal} векторов"
        )

    def load_community_index(self) -> None:
        """Загружает communities FAISS индекс и маппинг в память."""
        if self._communities_index is not None:
            return  # Уже загружен
        
        index_file = self.vectors_path / "communities.faiss"
        mapping_file = self.vectors_path / "communities_mapping.parquet"
        
        logger.info(f"Загрузка communities FAISS индекса из {index_file}...")
        self._communities_index = faiss.read_index(str(index_file))
        self._communities_mapping = pd.read_parquet(mapping_file)
        logger.info(
            f"Communities индекс загружен: {self._communities_index.ntotal} векторов"
        )

    def load_text_unit_index(self) -> None:
        """Загружает text_units FAISS индекс и маппинг в память."""
        if self._text_units_index is not None:
            return  # Уже загружен
        
        index_file = self.vectors_path / "text_units.faiss"
        mapping_file = self.vectors_path / "text_units_mapping.parquet"
        
        # Проверяем существование (text_units опциональны)
        if not index_file.exists():
            logger.info("Text units FAISS индекс не найден (опционально)")
            return
        
        logger.info(f"Загрузка text_units FAISS индекса из {index_file}...")
        self._text_units_index = faiss.read_index(str(index_file))
        self._text_units_mapping = pd.read_parquet(mapping_file)
        logger.info(
            f"Text units индекс загружен: {self._text_units_index.ntotal} векторов"
        )

    def search_entities(
        self,
        query_embedding: list[float],
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Поиск наиболее похожих entities.
        
        Args:
            query_embedding: Вектор запроса
            k: Количество результатов
        
        Returns:
            Список кортежей (entity_name, similarity_score)
        """
        if self._entities_index is None:
            self.load_entity_index()
        
        # Конвертируем в numpy
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Поиск в FAISS (L2 distance)
        distances, indices = self._entities_index.search(query_vector, k)  # type: ignore
        
        # Маппинг faiss_id -> entity_name
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS возвращает -1 если не нашел k результатов
                break
            
            entity_name = self._entities_mapping.loc[  # type: ignore
                self._entities_mapping["faiss_id"] == idx, "entity_name"  # type: ignore
            ].values[0]
            
            # Конвертируем L2 distance в similarity score (0-1)
            # Используем формулу: similarity = 1 / (1 + distance)
            similarity = 1.0 / (1.0 + float(dist))
            
            results.append((entity_name, similarity))
        
        return results

    def search_communities(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Поиск наиболее похожих communities.
        
        Args:
            query_embedding: Вектор запроса
            k: Количество результатов
        
        Returns:
            Список кортежей (community_id, similarity_score)
        """
        if self._communities_index is None:
            self.load_community_index()
        
        # Конвертируем в numpy
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Поиск в FAISS (L2 distance)
        distances, indices = self._communities_index.search(query_vector, k)  # type: ignore
        
        # Маппинг faiss_id -> community_id
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS возвращает -1 если не нашел k результатов
                break
            
            community_id = self._communities_mapping.loc[  # type: ignore
                self._communities_mapping["faiss_id"] == idx, "community_id"  # type: ignore
            ].values[0]
            
            # Конвертируем L2 distance в similarity score (0-1)
            similarity = 1.0 / (1.0 + float(dist))
            
            results.append((community_id, similarity))
        
        return results

    def get_entity_embedding(self, entity_name: str) -> list[float] | None:
        """
        Получает эмбеддинг entity по имени.
        
        Args:
            entity_name: Имя сущности
        
        Returns:
            Эмбеддинг или None если не найден
        """
        if self._entities_index is None:
            self.load_entity_index()
        
        # Находим faiss_id
        mask = self._entities_mapping["entity_name"] == entity_name  # type: ignore
        if not mask.any():
            return None
        
        faiss_id = self._entities_mapping.loc[mask, "faiss_id"].values[0]  # type: ignore
        
        # Извлекаем вектор из FAISS
        vector = self._entities_index.reconstruct(int(faiss_id))  # type: ignore
        return vector.tolist()

    def get_community_embedding(self, community_id: str) -> list[float] | None:
        """
        Получает эмбеддинг community по ID.
        
        Args:
            community_id: ID community
        
        Returns:
            Эмбеддинг или None если не найден
        """
        if self._communities_index is None:
            self.load_community_index()
        
        # Находим faiss_id
        mask = self._communities_mapping["community_id"] == community_id  # type: ignore
        if not mask.any():
            return None
        
        faiss_id = self._communities_mapping.loc[mask, "faiss_id"].values[0]  # type: ignore
        
        # Извлекаем вектор из FAISS
        vector = self._communities_index.reconstruct(int(faiss_id))  # type: ignore
        return vector.tolist()

    def search_text_units(
        self,
        query_embedding: list[float],
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Поиск наиболее похожих text_units (для классического RAG).
        
        Args:
            query_embedding: Вектор запроса
            k: Количество результатов
        
        Returns:
            Список кортежей (text_unit_id, similarity_score)
        """
        if self._text_units_index is None:
            self.load_text_unit_index()
        
        if self._text_units_index is None:
            logger.warning("Text units индекс не загружен")
            return []
        
        # Конвертируем в numpy
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Поиск в FAISS (L2 distance)
        distances, indices = self._text_units_index.search(query_vector, k)  # type: ignore
        
        # Маппинг faiss_id -> text_unit_id
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS возвращает -1 если не нашел k результатов
                break
            
            text_unit_id = self._text_units_mapping.loc[  # type: ignore
                self._text_units_mapping["faiss_id"] == idx, "text_unit_id"  # type: ignore
            ].values[0]
            
            # Конвертируем L2 distance в similarity score (0-1)
            similarity = 1.0 / (1.0 + float(dist))
            
            results.append((text_unit_id, similarity))
        
        return results

    def get_text_unit_embedding(self, text_unit_id: str) -> list[float] | None:
        """
        Получает эмбеддинг text_unit по ID.
        
        Args:
            text_unit_id: ID text_unit
        
        Returns:
            Эмбеддинг или None если не найден
        """
        if self._text_units_index is None:
            self.load_text_unit_index()
        
        if self._text_units_index is None or self._text_units_mapping is None:
            return None
        
        # Находим faiss_id
        mask = self._text_units_mapping["text_unit_id"] == text_unit_id  # type: ignore
        if not mask.any():
            return None
        
        faiss_id = self._text_units_mapping.loc[mask, "faiss_id"].values[0]  # type: ignore
        
        # Извлекаем вектор из FAISS
        vector = self._text_units_index.reconstruct(int(faiss_id))  # type: ignore
        return vector.tolist()

