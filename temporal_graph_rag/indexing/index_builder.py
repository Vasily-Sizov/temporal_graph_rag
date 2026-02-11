"""Основной модуль для построения индекса Temporal Graph RAG."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from temporal_graph_rag.indexing.community_builder import CommunityBuilder
from temporal_graph_rag.indexing.graph_extractor import GraphExtractor
from temporal_graph_rag.models.data_models import Community, Entity, Relationship, TextUnit
from temporal_graph_rag.settings import (
    CHECKPOINT_FREQUENCY,
    CREATE_COMMUNITY_EMBEDDINGS,
    CREATE_ENTITY_EMBEDDINGS,
    CREATE_TEXT_UNIT_EMBEDDINGS,
    ENABLE_CHECKPOINTS,
)
from temporal_graph_rag.storage.parquet_storage import ParquetStorage
from temporal_graph_rag.storage.vector_storage import VectorStorage
from temporal_graph_rag.utils.api_client import APIClient
from temporal_graph_rag.utils.chunk_loader import ChunkLoader

logger = logging.getLogger(__name__)


class IndexBuilder:
    """Строит полный индекс Temporal Graph RAG."""

    def __init__(
        self,
        api_client: APIClient,
        window_size: int = 20,
        overlap: int = 5,
    ) -> None:
        """
        Инициализация построителя индекса.

        Args:
            api_client: Клиент для работы с API
            window_size: Размер временного окна в чанках
            overlap: Перекрытие между окнами
        """
        self.api_client = api_client
        self.graph_extractor = GraphExtractor(api_client)
        self.community_builder = CommunityBuilder(
            api_client, window_size=window_size, overlap=overlap
        )

    def build_index(
        self,
        chunks_dir: str | Path,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        """
        Строит полный индекс из директории с чанками.

        Args:
            chunks_dir: Директория с JSON файлами чанков
            output_dir: Директория для сохранения индекса

        Returns:
            Статистика построения индекса
        """
        logger.info("=" * 80)
        logger.info("НАЧАЛО ПОСТРОЕНИЯ TEMPORAL GRAPH RAG ИНДЕКСА")
        logger.info("=" * 80)

        # Шаг 1: Загрузка чанков
        logger.info("\n[1/5] Загрузка чанков...")
        text_units = ChunkLoader.load_chunks_from_directory(chunks_dir)
        logger.info(f"Загружено {len(text_units)} текстовых единиц")

        # Группируем по книгам
        text_units_by_book = ChunkLoader.group_by_book(text_units)
        logger.info(f"Книг: {len(text_units_by_book)}")

        # Шаг 2: Извлечение графа
        logger.info("\n[2/5] Извлечение сущностей и отношений...")
        entities_dict, relationships_dict = self.graph_extractor.extract_from_text_units(
            text_units
        )
        logger.info(
            f"Извлечено: {len(entities_dict)} сущностей, "
            f"{len(relationships_dict)} отношений"
        )

        # Шаг 3: Построение temporal communities
        logger.info("\n[3/5] Построение temporal communities...")
        all_communities = []

        for book_number in sorted(text_units_by_book.keys()):
            book_units = text_units_by_book[book_number]
            logger.info(
                f"Обработка книги {book_number}: {len(book_units)} чанков"
            )

            communities = self.community_builder.create_temporal_communities(
                text_units=book_units,
                entities_dict=entities_dict,
                relationships_dict=relationships_dict,
            )

            all_communities.extend(communities)

        logger.info(f"Создано {len(all_communities)} temporal communities")

        # Шаг 4: Создание эмбеддингов
        logger.info("\n[4/5] Создание эмбеддингов...")
        self._create_embeddings(text_units, all_communities, entities_dict)

        # Шаг 5: Сохранение индекса
        logger.info("\n[5/5] Сохранение индекса...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stats = self._save_index(
            output_path=output_path,
            text_units=text_units,
            entities_dict=entities_dict,
            relationships_dict=relationships_dict,
            communities=all_communities,
        )

        logger.info("\n" + "=" * 80)
        logger.info("ИНДЕКС УСПЕШНО ПОСТРОЕН")
        logger.info("=" * 80)
        logger.info(f"Текстовых единиц: {stats['text_units']}")
        logger.info(f"Сущностей: {stats['entities']}")
        logger.info(f"Отношений: {stats['relationships']}")
        logger.info(f"Communities: {stats['communities']}")
        logger.info(f"Директория: {output_path}")
        logger.info("=" * 80)

        return stats

    def _create_embeddings(
        self,
        text_units: list[TextUnit],
        communities: list[Community],
        entities_dict: dict[str, Entity],
    ) -> None:
        """
        Создает эмбеддинги для text_units, communities и entities (опционально).
        
        ВАЖНО: Использует embed_single() для каждого текста, так как
        embedder сервер настроен на --max-num-seqs=1 и может обрабатывать
        только 1 запрос одновременно.

        Args:
            text_units: Список текстовых единиц
            communities: Список сообществ
            entities_dict: Словарь сущностей
        """
        # Эмбеддинги для text_units (опционально для сравнения с классическим RAG)
        if CREATE_TEXT_UNIT_EMBEDDINGS:
            logger.info(
                f"Создание эмбеддингов для {len(text_units)} text_units "
                "(для сравнения с классическим RAG)..."
            )
            
            for idx, unit in enumerate(text_units, 1):
                try:
                    # Используем contextualized_text для более качественных эмбеддингов
                    unit.embedding = self.api_client.embed_single(
                        unit.contextualized_text or unit.text
                    )
                    
                    if idx % 50 == 0:
                        logger.info(f"Обработано {idx}/{len(text_units)} text_units")
                except Exception as e:
                    logger.error(f"Ошибка создания эмбеддинга для text_unit {unit.id}: {e}")
            
            logger.info(f"Создано {len(text_units)} эмбеддингов для text_units")
        else:
            logger.info("Пропуск создания эмбеддингов для text_units (CREATE_TEXT_UNIT_EMBEDDINGS=False)")
        
        # Эмбеддинги для community reports
        if CREATE_COMMUNITY_EMBEDDINGS:
            logger.info(f"Создание эмбеддингов для {len(communities)} community reports...")
            
            for idx, comm in enumerate(communities, 1):
                try:
                    comm.embedding = self.api_client.embed_single(comm.report)
                    if idx % 10 == 0:
                        logger.info(f"Обработано {idx}/{len(communities)} communities")
                except Exception as e:
                    logger.error(f"Ошибка создания эмбеддинга для community {comm.id}: {e}")
            
            logger.info(f"Создано {len(communities)} эмбеддингов для communities")
        else:
            logger.info("Пропуск создания эмбеддингов для communities (CREATE_COMMUNITY_EMBEDDINGS=False)")

        # Эмбеддинги для сущностей
        if CREATE_ENTITY_EMBEDDINGS:
            logger.info(f"Создание эмбеддингов для {len(entities_dict)} сущностей...")
            
            entities_list = list(entities_dict.values())
            for idx, entity in enumerate(entities_list, 1):
                try:
                    entity_text = f"{entity.name}: {entity.description}"
                    entity.embedding = self.api_client.embed_single(entity_text)
                    
                    if idx % 100 == 0:
                        logger.info(f"Обработано {idx}/{len(entities_list)} сущностей")
                except Exception as e:
                    logger.error(f"Ошибка создания эмбеддинга для сущности {entity.name}: {e}")
            
            logger.info(f"Создано {len(entities_list)} эмбеддингов для сущностей")
        else:
            logger.info("Пропуск создания эмбеддингов для entities (CREATE_ENTITY_EMBEDDINGS=False)")

    def _save_index(
        self,
        output_path: Path,
        text_units: list[TextUnit],
        entities_dict: dict[str, Entity],
        relationships_dict: dict[str, Relationship],
        communities: list[Community],
    ) -> dict[str, int]:
        """
        Сохраняет индекс в Parquet + FAISS.

        Args:
            output_path: Путь для сохранения
            text_units: Список текстовых единиц
            entities_dict: Словарь сущностей
            relationships_dict: Словарь отношений
            communities: Список сообществ

        Returns:
            Статистика сохранения
        """
        logger.info("Сохранение индекса в Parquet + FAISS...")
        
        # Инициализируем хранилища
        parquet_storage = ParquetStorage(output_path)
        vector_storage = VectorStorage(output_path)
        
        # Сохраняем табличные данные в Parquet
        parquet_storage.save_text_units(text_units)
        parquet_storage.save_entities(entities_dict)
        parquet_storage.save_relationships(relationships_dict)
        parquet_storage.save_communities(communities)
        
        # Сохраняем векторы в FAISS
        # Text units (опционально)
        text_units_with_embeddings = [
            unit for unit in text_units if unit.embedding is not None
        ]
        
        if text_units_with_embeddings:
            text_unit_ids = [unit.id for unit in text_units_with_embeddings]
            text_unit_embeddings = [unit.embedding for unit in text_units_with_embeddings]
            vector_storage.save_text_unit_vectors(text_unit_ids, text_unit_embeddings)
        else:
            if CREATE_TEXT_UNIT_EMBEDDINGS:
                logger.warning("Нет text_units с эмбеддингами для сохранения в FAISS")
        
        # Собираем entities с эмбеддингами
        entities_with_embeddings = [
            (name, entity)
            for name, entity in entities_dict.items()
            if entity.embedding is not None
        ]
        
        if entities_with_embeddings:
            entity_names = [name for name, _ in entities_with_embeddings]
            entity_embeddings = [
                entity.embedding for _, entity in entities_with_embeddings
            ]
            vector_storage.save_entity_vectors(entity_names, entity_embeddings)
        else:
            if CREATE_ENTITY_EMBEDDINGS:
                logger.warning("Нет entities с эмбеддингами для сохранения в FAISS")
        
        # Собираем communities с эмбеддингами
        communities_with_embeddings = [
            comm for comm in communities if comm.embedding is not None
        ]
        
        if communities_with_embeddings:
            community_ids = [comm.id for comm in communities_with_embeddings]
            community_embeddings = [comm.embedding for comm in communities_with_embeddings]
            vector_storage.save_community_vectors(community_ids, community_embeddings)
        else:
            if CREATE_COMMUNITY_EMBEDDINGS:
                logger.warning("Нет communities с эмбеддингами для сохранения в FAISS")
        
        # Сохраняем метаданные
        metadata = {
            "version": "1.0",
            "embedding_dim": (
                len(text_unit_embeddings[0])
                if text_units_with_embeddings
                else (
                    len(entity_embeddings[0])
                    if entities_with_embeddings
                    else (
                        len(community_embeddings[0]) if communities_with_embeddings else 0
                    )
                )
            ),
            "total_text_units": len(text_units),
            "total_entities": len(entities_dict),
            "total_relationships": len(relationships_dict),
            "total_communities": len(communities),
            "text_units_with_embeddings": len(text_units_with_embeddings),
            "entities_with_embeddings": len(entities_with_embeddings),
            "communities_with_embeddings": len(communities_with_embeddings),
            "create_text_unit_embeddings": CREATE_TEXT_UNIT_EMBEDDINGS,
            "create_entity_embeddings": CREATE_ENTITY_EMBEDDINGS,
            "create_community_embeddings": CREATE_COMMUNITY_EMBEDDINGS,
        }
        
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info("Индекс успешно сохранен в Parquet + FAISS")
        
        return {
            "text_units": len(text_units),
            "entities": len(entities_dict),
            "relationships": len(relationships_dict),
            "communities": len(communities),
        }

    @staticmethod
    def _text_unit_to_dict(tu: TextUnit) -> dict[str, Any]:
        """Конвертирует TextUnit в словарь."""
        return {
            "id": tu.id,
            "text": tu.text,
            "contextualized_text": tu.contextualized_text,
            "temporal_position": {
                "book_number": tu.temporal_position.book_number,
                "book_title": tu.temporal_position.book_title,
                "chunk_index": tu.temporal_position.chunk_index,
                "global_chunk_index": tu.temporal_position.global_chunk_index,
                "relative_position": tu.temporal_position.relative_position,
            },
            "text_tokens": tu.text_tokens,
            "metadata": tu.metadata,
            "entities": tu.entities,
            "relationships": tu.relationships,
        }

    @staticmethod
    def _entity_to_dict(entity: Entity) -> dict[str, Any]:
        """Конвертирует Entity в словарь."""
        return {
            "name": entity.name,
            "type": entity.type,
            "book_number": entity.book_number,
            "chapter": entity.chapter,
            "description": entity.description,
            "descriptions_raw": entity.descriptions_raw,
            "text_unit_ids": entity.text_unit_ids,
            "embedding": entity.embedding,
        }

    @staticmethod
    def _relationship_to_dict(rel: Relationship) -> dict[str, Any]:
        """Конвертирует Relationship в словарь."""
        return {
            "source": rel.source,
            "target": rel.target,
            "book_number": rel.book_number,
            "chapter": rel.chapter,
            "description": rel.description,
            "descriptions_raw": rel.descriptions_raw,
            "type": rel.type,
            "weight": rel.weight,
            "text_unit_ids": rel.text_unit_ids,
        }

    @staticmethod
    def _community_to_dict(comm: Community) -> dict[str, Any]:
        """Конвертирует Community в словарь."""
        return {
            "id": comm.id,
            "book_number": comm.book_number,
            "temporal_range": comm.temporal_range,
            "relative_position_range": comm.relative_position_range,
            "title": comm.title,
            "summary": comm.summary,
            "report": comm.report,
            "entities": comm.entities,
            "relationships": comm.relationships,
            "text_unit_ids": comm.text_unit_ids,
            "embedding": comm.embedding,
        }

    async def build_index_async(
        self, chunks_dir: str, output_dir: str
    ) -> dict[str, int]:
        """
        АСИНХРОННО строит полный индекс из директории с чанками.
        
        Использует параллельные запросы к LLM (до 8 одновременно) для:
        - Извлечения сущностей и отношений
        - Генерации community reports
        
        Args:
            chunks_dir: Директория с JSON файлами чанков
            output_dir: Директория для сохранения индекса

        Returns:
            Статистика построения индекса
        """
        logger.info("=" * 80)
        logger.info("НАЧАЛО АСИНХРОННОГО ПОСТРОЕНИЯ TEMPORAL GRAPH RAG ИНДЕКСА")
        if ENABLE_CHECKPOINTS:
            logger.info("СИСТЕМА CHECKPOINTS ВКЛЮЧЕНА")
        logger.info("=" * 80)

        # Подготовка директории для промежуточных результатов
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        intermediate_dir = output_path / "intermediate"
        intermediate_dir.mkdir(exist_ok=True)

        # Проверяем наличие checkpoints
        completed_steps = self._check_completed_steps(intermediate_dir)
        logger.info(f"\nСтатус шагов: {completed_steps}")

        # Шаг 1: Загрузка чанков
        if completed_steps["step1_text_units"] and ENABLE_CHECKPOINTS:
            logger.info("\n[1/5] Загрузка чанков (из checkpoint)...")
            text_units = self._load_step1_checkpoint(intermediate_dir)
        else:
            logger.info("\n[1/5] Загрузка чанков...")
            text_units = ChunkLoader.load_chunks_from_directory(chunks_dir)
            logger.info(f"Загружено {len(text_units)} текстовых единиц")

            # Группируем по книгам
            text_units_by_book = ChunkLoader.group_by_book(text_units)
            logger.info(f"Книг: {len(text_units_by_book)}")
            
            # Сохраняем промежуточный результат: text_units
            if ENABLE_CHECKPOINTS:
                logger.info("Сохранение промежуточного результата: text_units...")
                self._save_text_units_intermediate(intermediate_dir, text_units)

        # Шаг 2: АСИНХРОННОЕ извлечение графа
        if completed_steps["step2_graph"] and ENABLE_CHECKPOINTS:
            logger.info("\n[2/5] Извлечение сущностей и отношений (из checkpoint)...")
            entities_dict, relationships_dict = self._load_step2_checkpoint(
                intermediate_dir
            )
        else:
            logger.info("\n[2/5] Извлечение сущностей и отношений (АСИНХРОННО)...")
            logger.info(f"Checkpoint frequency: каждые {CHECKPOINT_FREQUENCY} чанков")
            
            entities_dict, relationships_dict = await self.graph_extractor.extract_from_text_units_async(
                text_units=text_units,
                checkpoint_dir=intermediate_dir if ENABLE_CHECKPOINTS else None,
                checkpoint_frequency=CHECKPOINT_FREQUENCY,
            )
            logger.info(
                f"Извлечено: {len(entities_dict)} сущностей, "
                f"{len(relationships_dict)} отношений"
            )
            
            # Сохраняем финальный результат: граф (entities + relationships)
            if ENABLE_CHECKPOINTS:
                logger.info("Сохранение финального результата: граф...")
                self._save_graph_intermediate(
                    intermediate_dir, entities_dict, relationships_dict
                )

        # Шаг 3: АСИНХРОННОЕ построение temporal communities
        if completed_steps["step3_communities"] and ENABLE_CHECKPOINTS:
            logger.info("\n[3/5] Построение temporal communities (из checkpoint)...")
            all_communities = self._load_step3_checkpoint(intermediate_dir)
        else:
            logger.info("\n[3/5] Построение temporal communities (АСИНХРОННО)...")
            all_communities = []
            
            text_units_by_book = ChunkLoader.group_by_book(text_units)

            for book_number in sorted(text_units_by_book.keys()):
                book_units = text_units_by_book[book_number]
                logger.info(
                    f"Обработка книги {book_number}: {len(book_units)} чанков"
                )

                communities = await self.community_builder.create_temporal_communities_async(
                    text_units=book_units,
                    entities_dict=entities_dict,
                    relationships_dict=relationships_dict,
                )

                all_communities.extend(communities)

            logger.info(f"Создано {len(all_communities)} temporal communities")
            
            # Сохраняем промежуточный результат: communities (без эмбеддингов)
            if ENABLE_CHECKPOINTS:
                logger.info("Сохранение промежуточного результата: communities...")
                self._save_communities_intermediate(
                    intermediate_dir, all_communities
                )

        # Шаг 4: Создание эмбеддингов (АСИНХРОННО)
        if completed_steps["step4_embeddings"] and ENABLE_CHECKPOINTS:
            logger.info("\n[4/5] Создание эмбеддингов (из checkpoint)...")
            # Загружаем communities и entities с эмбеддингами
            all_communities = self._load_step3_checkpoint(intermediate_dir)
            # У нас уже есть entities_dict, просто загружаем их эмбеддинги
            
            entities_with_emb_file = intermediate_dir / "step4_entities_with_embeddings.json"
            with open(entities_with_emb_file, "r", encoding="utf-8") as f:
                entities_data = json.load(f)
            
            # Обновляем эмбеддинги в существующих entities
            for item in entities_data:
                if item["name"] in entities_dict and item.get("embedding"):
                    entities_dict[item["name"]].embedding = item["embedding"]
            
            # Загружаем communities с эмбеддингами
            communities_with_emb_file = intermediate_dir / "step4_communities_with_embeddings.json"
            with open(communities_with_emb_file, "r", encoding="utf-8") as f:
                communities_data = json.load(f)
            
            for comm, item in zip(all_communities, communities_data):
                if item.get("embedding"):
                    comm.embedding = item["embedding"]
            
            logger.info("✓ Эмбеддинги загружены из checkpoint")
        else:
            logger.info("\n[4/5] Создание эмбеддингов (АСИНХРОННО)...")
            await self._create_embeddings_async(text_units, all_communities, entities_dict)
            
            # Сохраняем промежуточный результат: эмбеддинги
            if ENABLE_CHECKPOINTS:
                logger.info("Сохранение промежуточного результата: эмбеддинги...")
                self._save_embeddings_intermediate(
                    intermediate_dir, all_communities, entities_dict
                )

        # Шаг 5: Сохранение индекса
        logger.info("\n[5/5] Сохранение индекса...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stats = self._save_index(
            output_path=output_path,
            text_units=text_units,
            entities_dict=entities_dict,
            relationships_dict=relationships_dict,
            communities=all_communities,
        )

        logger.info("\n" + "=" * 80)
        logger.info("ИНДЕКС УСПЕШНО ПОСТРОЕН")
        logger.info("=" * 80)
        logger.info(f"Текстовых единиц: {stats['text_units']}")
        logger.info(f"Сущностей: {stats['entities']}")
        logger.info(f"Отношений: {stats['relationships']}")
        logger.info(f"Communities: {stats['communities']}")
        logger.info(f"Директория: {output_path}")
        logger.info("=" * 80)

        return stats

    async def _create_embeddings_async(
        self,
        text_units: list[TextUnit],
        communities: list[Community],
        entities_dict: dict[str, Entity],
    ) -> None:
        """
        Асинхронно создает эмбеддинги для text_units, communities и сущностей.
        
        ВАЖНО: Embedder имеет --max-num-seqs=1, поэтому запросы будут
        выполняться последовательно, несмотря на асинхронность.

        Args:
            text_units: Список текстовых единиц
            communities: Список сообществ
            entities_dict: Словарь сущностей
        """
        # Эмбеддинги для text_units (опционально)
        if CREATE_TEXT_UNIT_EMBEDDINGS:
            logger.info(f"Создание эмбеддингов для {len(text_units)} text_units...")
            
            text_unit_texts = [
                unit.contextualized_text or unit.text
                for unit in text_units
            ]
            embeddings = await self.api_client.embed_batch_async(
                texts=text_unit_texts,
                show_progress=True,
            )
            
            for unit, emb in zip(text_units, embeddings):
                unit.embedding = emb
            
            logger.info(f"Создано {len(embeddings)} эмбеддингов для text_units")
        else:
            logger.info("Пропуск создания эмбеддингов для text_units (CREATE_TEXT_UNIT_EMBEDDINGS=False)")
        
        # Эмбеддинги для community reports
        if CREATE_COMMUNITY_EMBEDDINGS:
            logger.info(f"Создание эмбеддингов для {len(communities)} community reports...")
            
            community_texts = [comm.report for comm in communities]
            embeddings = await self.api_client.embed_batch_async(
                texts=community_texts,
                show_progress=True,
            )
            
            for comm, emb in zip(communities, embeddings):
                comm.embedding = emb
            
            logger.info(f"Создано {len(embeddings)} эмбеддингов для communities")
        else:
            logger.info("Пропуск создания эмбеддингов для communities (CREATE_COMMUNITY_EMBEDDINGS=False)")

        # Эмбеддинги для сущностей
        if CREATE_ENTITY_EMBEDDINGS:
            logger.info(f"Создание эмбеддингов для {len(entities_dict)} сущностей...")
            
            entities_list = list(entities_dict.values())
            entity_texts = [
                f"{entity.name}: {entity.description}"
                for entity in entities_list
            ]
            
            embeddings = await self.api_client.embed_batch_async(
                texts=entity_texts,
                show_progress=True,
            )
            
            for entity, emb in zip(entities_list, embeddings):
                entity.embedding = emb
            
            logger.info(f"Создано {len(embeddings)} эмбеддингов для сущностей")
        else:
            logger.info("Пропуск создания эмбеддингов для сущностей (CREATE_ENTITY_EMBEDDINGS=False)")

    def _save_text_units_intermediate(
        self,
        intermediate_dir: Path,
        text_units: list[TextUnit],
    ) -> None:
        """
        Сохраняет промежуточный результат: загруженные text_units (Parquet).
        
        Args:
            intermediate_dir: Директория для промежуточных файлов
            text_units: Список текстовых единиц
        """
        # Используем ParquetStorage
        temp_storage = ParquetStorage(intermediate_dir.parent)
        temp_storage.data_path = intermediate_dir
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        temp_storage.save_text_units(text_units)
        
        # Переименовываем в step1
        (intermediate_dir / "text_units.parquet").rename(
            intermediate_dir / "step1_text_units.parquet"
        )
        logger.info(f"✓ Сохранено {len(text_units)} text_units -> step1_text_units.parquet")

    def _save_graph_intermediate(
        self,
        intermediate_dir: Path,
        entities_dict: dict[str, Entity],
        relationships_dict: dict[str, Relationship],
    ) -> None:
        """
        Сохраняет промежуточный результат: граф (entities + relationships) (Parquet).
        
        Args:
            intermediate_dir: Директория для промежуточных файлов
            entities_dict: Словарь сущностей
            relationships_dict: Словарь отношений
        """
        # Используем ParquetStorage
        temp_storage = ParquetStorage(intermediate_dir.parent)
        temp_storage.data_path = intermediate_dir
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем entities
        temp_storage.save_entities(entities_dict)
        (intermediate_dir / "entities.parquet").rename(
            intermediate_dir / "step2_entities.parquet"
        )
        logger.info(f"✓ Сохранено {len(entities_dict)} entities -> step2_entities.parquet")
        
        # Сохраняем relationships
        temp_storage.save_relationships(relationships_dict)
        (intermediate_dir / "relationships.parquet").rename(
            intermediate_dir / "step2_relationships.parquet"
        )
        logger.info(f"✓ Сохранено {len(relationships_dict)} relationships -> step2_relationships.parquet")

    def _save_communities_intermediate(
        self,
        intermediate_dir: Path,
        communities: list[Community],
    ) -> None:
        """
        Сохраняет промежуточный результат: communities (без эмбеддингов) (Parquet).
        
        Args:
            intermediate_dir: Директория для промежуточных файлов
            communities: Список communities
        """
        # Используем ParquetStorage
        temp_storage = ParquetStorage(intermediate_dir.parent)
        temp_storage.data_path = intermediate_dir
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        temp_storage.save_communities(communities)
        (intermediate_dir / "communities.parquet").rename(
            intermediate_dir / "step3_communities_no_embeddings.parquet"
        )
        logger.info(f"✓ Сохранено {len(communities)} communities -> step3_communities_no_embeddings.parquet")

    def _save_embeddings_intermediate(
        self,
        intermediate_dir: Path,
        communities: list[Community],
        entities_dict: dict[str, Entity],
    ) -> None:
        """
        Сохраняет промежуточный результат: эмбеддинги для communities и entities (Parquet).
        
        Args:
            intermediate_dir: Директория для промежуточных файлов
            communities: Список communities с эмбеддингами
            entities_dict: Словарь сущностей с эмбеддингами
        """
        # Используем ParquetStorage
        temp_storage = ParquetStorage(intermediate_dir.parent)
        temp_storage.data_path = intermediate_dir
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем communities с эмбеддингами
        temp_storage.save_communities(communities)
        (intermediate_dir / "communities.parquet").rename(
            intermediate_dir / "step4_communities_with_embeddings.parquet"
        )
        logger.info(f"✓ Сохранено {len(communities)} communities с эмбеддингами -> step4_communities_with_embeddings.parquet")
        
        # Сохраняем entities с эмбеддингами
        temp_storage.save_entities(entities_dict)
        (intermediate_dir / "entities.parquet").rename(
            intermediate_dir / "step4_entities_with_embeddings.parquet"
        )
        logger.info(f"✓ Сохранено {len(entities_dict)} entities с эмбеддингами -> step4_entities_with_embeddings.parquet")

    def _check_completed_steps(self, intermediate_dir: Path) -> dict[str, bool]:
        """
        Проверяет какие шаги уже завершены.
        
        Args:
            intermediate_dir: Директория с промежуточными файлами
            
        Returns:
            Словарь {step_name: completed}
        """
        return {
            "step1_text_units": (intermediate_dir / "step1_text_units.parquet").exists(),
            "step2_graph": (
                (intermediate_dir / "step2_entities.parquet").exists()
                and (intermediate_dir / "step2_relationships.parquet").exists()
            ),
            "step3_communities": (
                intermediate_dir / "step3_communities_no_embeddings.parquet"
            ).exists(),
            "step4_embeddings": (
                (intermediate_dir / "step4_communities_with_embeddings.parquet").exists()
                and (intermediate_dir / "step4_entities_with_embeddings.parquet").exists()
            ),
        }

    def _load_step1_checkpoint(self, intermediate_dir: Path) -> list[TextUnit]:
        """Загружает checkpoint шага 1 (text_units из Parquet)."""
        logger.info(f"Загрузка checkpoint: step1_text_units.parquet")
        
        # Используем ParquetStorage
        temp_storage = ParquetStorage(intermediate_dir.parent)
        temp_storage.data_path = intermediate_dir
        
        # Временно переименовываем для загрузки
        checkpoint_file = intermediate_dir / "step1_text_units.parquet"
        temp_file = intermediate_dir / "text_units.parquet"
        checkpoint_file.rename(temp_file)
        
        try:
            text_units = temp_storage.load_text_units()
            logger.info(f"✓ Загружено {len(text_units)} text_units из checkpoint")
            return text_units
        finally:
            # Возвращаем имя обратно
            temp_file.rename(checkpoint_file)

    def _load_step2_checkpoint(
        self, intermediate_dir: Path
    ) -> tuple[dict[str, Entity], dict[str, Relationship]]:
        """Загружает checkpoint шага 2 (entities + relationships из Parquet)."""
        logger.info(f"Загрузка checkpoint: step2_entities.parquet и step2_relationships.parquet")
        
        # Используем ParquetStorage
        temp_storage = ParquetStorage(intermediate_dir.parent)
        temp_storage.data_path = intermediate_dir
        
        # Загружаем entities
        entities_checkpoint = intermediate_dir / "step2_entities.parquet"
        entities_temp = intermediate_dir / "entities.parquet"
        entities_checkpoint.rename(entities_temp)
        
        try:
            entities_dict = temp_storage.load_entities()
        finally:
            entities_temp.rename(entities_checkpoint)
        
        # Загружаем relationships
        relationships_checkpoint = intermediate_dir / "step2_relationships.parquet"
        relationships_temp = intermediate_dir / "relationships.parquet"
        relationships_checkpoint.rename(relationships_temp)
        
        try:
            relationships_dict = temp_storage.load_relationships()
        finally:
            relationships_temp.rename(relationships_checkpoint)
        
        logger.info(
            f"✓ Загружено {len(entities_dict)} entities и {len(relationships_dict)} relationships из checkpoint"
        )
        return entities_dict, relationships_dict

    def _load_step3_checkpoint(self, intermediate_dir: Path) -> list[Community]:
        """Загружает checkpoint шага 3 (communities без эмбеддингов из Parquet)."""
        logger.info(f"Загрузка checkpoint: step3_communities_no_embeddings.parquet")
        
        # Используем ParquetStorage
        temp_storage = ParquetStorage(intermediate_dir.parent)
        temp_storage.data_path = intermediate_dir
        
        # Временно переименовываем для загрузки
        checkpoint_file = intermediate_dir / "step3_communities_no_embeddings.parquet"
        temp_file = intermediate_dir / "communities.parquet"
        checkpoint_file.rename(temp_file)
        
        try:
            communities = temp_storage.load_communities()
            logger.info(f"✓ Загружено {len(communities)} communities из checkpoint")
            return communities
        finally:
            # Возвращаем имя обратно
            temp_file.rename(checkpoint_file)

