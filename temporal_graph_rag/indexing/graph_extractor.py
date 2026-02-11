"""Извлечение сущностей и отношений из текстовых единиц."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from temporal_graph_rag.models.data_models import (
    Entity,
    Relationship,
    TextUnit,
)
from temporal_graph_rag.prompts import EXTRACT_GRAPH_PROMPT
from temporal_graph_rag.settings import (
    EXTRACT_GRAPH_MAX_TOKENS,
    EXTRACT_GRAPH_TEMPERATURE,
)
from temporal_graph_rag.storage.parquet_storage import ParquetStorage
from temporal_graph_rag.utils.api_client import APIClient
from temporal_graph_rag.utils.id_utils import create_relationship_id

logger = logging.getLogger(__name__)


class GraphExtractor:
    """Извлекает сущности и отношения из текстовых единиц."""

    def __init__(self, api_client: APIClient) -> None:
        """
        Инициализация экстрактора графа.

        Args:
            api_client: Клиент для работы с API
        """
        self.api_client = api_client

    def extract_from_text_unit(
        self, text_unit: TextUnit
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Извлекает сущности и отношения из одной текстовой единицы.

        Args:
            text_unit: Текстовая единица для анализа

        Returns:
            Кортеж (entities, relationships) в виде словарей
        """
        # Формируем промпт с временным контекстом
        prompt = EXTRACT_GRAPH_PROMPT.format(
            book_title=text_unit.temporal_position.book_title,
            book_number=text_unit.temporal_position.book_number,
            relative_position=text_unit.temporal_position.relative_position,
            text=text_unit.text,
        )

        try:
            # Генерируем ответ
            response = self.api_client.generate(
                prompt=prompt,
                temperature=EXTRACT_GRAPH_TEMPERATURE,
                max_tokens=EXTRACT_GRAPH_MAX_TOKENS,
            )

            # Парсим JSON
            # Убираем markdown форматирование если есть
            response = response.strip()
            if response.startswith("```"):
                # Убираем ```json и ```
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            data = json.loads(response)

            entities = data.get("entities", [])
            relationships = data.get("relationships", [])

            logger.debug(
                f"Извлечено из {text_unit.id}: "
                f"{len(entities)} сущностей, {len(relationships)} отношений"
            )

            return entities, relationships

        except json.JSONDecodeError as e:
            logger.error(
                f"Ошибка парсинга JSON для {text_unit.id}: {e}\n"
                f"Ответ: {response[:500]}"
            )
            return [], []
        except Exception as e:
            logger.error(f"Ошибка извлечения графа для {text_unit.id}: {e}")
            return [], []

    def extract_from_text_units(
        self, text_units: list[TextUnit], batch_size: int = 1
    ) -> tuple[dict[str, Entity], dict[str, Relationship]]:
        """
        Извлекает и агрегирует сущности и отношения из списка текстовых единиц.
        
        УСТАРЕЛ: Используйте extract_from_text_units_async() для параллельной обработки.

        Args:
            text_units: Список текстовых единиц
            batch_size: Размер батча (пока не используется)

        Returns:
            Кортеж (entities_dict, relationships_dict)
        """
        logger.warning(
            "Используется синхронная версия extract_from_text_units(). "
            "Рекомендуется использовать extract_from_text_units_async() для ускорения."
        )
        
        logger.info(f"Начинаем извлечение графа из {len(text_units)} текстовых единиц")

        entities_dict: dict[str, Entity] = {}
        relationships_dict: dict[str, Relationship] = {}

        for idx, text_unit in enumerate(text_units, 1):
            if idx % 10 == 0:
                logger.info(f"Обработано {idx}/{len(text_units)} текстовых единиц")

            entities_raw, relationships_raw = self.extract_from_text_unit(text_unit)

            # Получаем контекст из text_unit
            book_number = text_unit.temporal_position.book_number
            headings = text_unit.metadata.get("headings", [])
            chapter = headings[0] if headings else "Unknown"

            # Обрабатываем сущности
            for entity_data in entities_raw:
                entity_name = entity_data.get("name", "").strip()
                if not entity_name:
                    continue

                # Ключ с book+chapter для уникальности
                entity_key = f"{entity_name}|{book_number}|{chapter}"

                # Создаем или обновляем сущность
                if entity_key not in entities_dict:
                    entities_dict[entity_key] = Entity(
                        name=entity_name,
                        type=entity_data.get("type", "UNKNOWN"),
                        book_number=book_number,
                        chapter=chapter,
                        description=entity_data.get("description", ""),
                        text_unit_ids=[],
                    )

                # Добавляем текстовую единицу
                entities_dict[entity_key].text_unit_ids.append(text_unit.id)

                # Добавляем в список сущностей текстовой единицы
                text_unit.entities.append(entity_key)

            # Обрабатываем отношения
            for rel_data in relationships_raw:
                source = rel_data.get("source", "").strip()
                target = rel_data.get("target", "").strip()

                if not source or not target:
                    continue

                # Ключи с book+chapter
                source_key = f"{source}|{book_number}|{chapter}"
                target_key = f"{target}|{book_number}|{chapter}"

                # Проверяем, что сущности существуют
                if source_key not in entities_dict or target_key not in entities_dict:
                    continue

                # Генерируем MD5 ID (как в GraphRAG)
                rel_id = create_relationship_id(source, target, book_number, chapter)

                # Создаем или обновляем отношение
                if rel_id not in relationships_dict:
                    relationships_dict[rel_id] = Relationship(
                        source=source,
                        target=target,
                        book_number=book_number,
                        chapter=chapter,
                        description=rel_data.get("description", ""),
                        type=rel_data.get("type", "UNKNOWN"),
                        weight=rel_data.get("strength", 5) / 10.0,
                        text_unit_ids=[],
                    )

                # Добавляем текстовую единицу
                relationships_dict[rel_id].text_unit_ids.append(text_unit.id)

                # Добавляем в список отношений текстовой единицы
                text_unit.relationships.append(rel_id)

        logger.info(
            f"Извлечение завершено: {len(entities_dict)} сущностей, "
            f"{len(relationships_dict)} отношений"
        )

        return entities_dict, relationships_dict

    async def extract_from_text_units_async(
        self,
        text_units: list[TextUnit],
        checkpoint_dir: Path | None = None,
        checkpoint_frequency: int = 20,
    ) -> tuple[dict[str, Entity], dict[str, Relationship]]:
        """
        Асинхронно извлекает сущности и отношения из текстовых единиц с поддержкой checkpoints.
        
        Использует параллельные запросы к LLM (до 8 одновременно) для ускорения.
        Сохраняет промежуточные результаты каждые checkpoint_frequency чанков.

        Args:
            text_units: Список текстовых единиц
            checkpoint_dir: Директория для checkpoints (если None - без checkpoints)
            checkpoint_frequency: Частота сохранения checkpoints (каждые N чанков)

        Returns:
            Кортеж (entities_dict, relationships_dict)
        """
        total = len(text_units)
        logger.info(
            f"Начинаем АСИНХРОННОЕ извлечение графа из {total} текстовых единиц "
            f"(до 8 параллельных запросов к LLM)"
        )

        # Инициализируем словари
        entities_dict: dict[str, Entity] = {}
        relationships_dict: dict[str, Relationship] = {}
        start_idx = 0

        # Загружаем checkpoint если есть
        if checkpoint_dir:
            entities_checkpoint = checkpoint_dir / "step2_checkpoint_entities.parquet"
            relationships_checkpoint = checkpoint_dir / "step2_checkpoint_relationships.parquet"
            
            if entities_checkpoint.exists() and relationships_checkpoint.exists():
                logger.info(f"Найден checkpoint: {checkpoint_dir}")
                entities_dict, relationships_dict, start_idx = self._load_graph_checkpoint(
                    checkpoint_dir, text_units
                )
                logger.info(
                    f"✓ Checkpoint загружен: {len(entities_dict)} entities, "
                    f"{len(relationships_dict)} relationships, "
                    f"обработано {start_idx}/{total} чанков"
                )
        
        if start_idx >= total:
            logger.info("Все чанки уже обработаны!")
            return entities_dict, relationships_dict
        
        logger.info(f"Начинаем с чанка {start_idx + 1}/{total}")

        # Обрабатываем оставшиеся чанки порциями
        for batch_start in range(start_idx, total, checkpoint_frequency):
            batch_end = min(batch_start + checkpoint_frequency, total)
            batch_units = text_units[batch_start:batch_end]
            
            logger.info(f"Обработка чанков [{batch_start + 1}:{batch_end}] / {total}...")
            
            # Генерируем промпты для батча
            prompts = []
            for text_unit in batch_units:
                prompt = EXTRACT_GRAPH_PROMPT.format(
                    book_title=text_unit.temporal_position.book_title,
                    book_number=text_unit.temporal_position.book_number,
                    relative_position=text_unit.temporal_position.relative_position,
                    text=text_unit.text,
                )
                prompts.append(prompt)

            # Параллельная генерация
            responses = await self.api_client.generate_batch_async(
                prompts=prompts,
                temperature=EXTRACT_GRAPH_TEMPERATURE,
                max_tokens=EXTRACT_GRAPH_MAX_TOKENS,
                show_progress=False,  # Не показываем внутренний прогресс
            )

            # Парсинг ответов и агрегация
            for text_unit, response in zip(batch_units, responses):
                try:
                    # Убираем markdown форматирование если есть
                    response = response.strip()
                    if response.startswith("```"):
                        lines = response.split("\n")
                        response = "\n".join(lines[1:-1])

                    data = json.loads(response)
                    entities_raw = data.get("entities", [])
                    relationships_raw = data.get("relationships", [])

                    # Получаем главу из metadata
                    # Теперь headings уже отфильтрованы в чанкере (только настоящие главы),
                    # поэтому берем первый heading как есть, включая "Annotation"
                    headings = text_unit.metadata.get("headings", [])
                    chapter = headings[0] if headings else "Unknown"
                    book_number = text_unit.temporal_position.book_number

                    # Обрабатываем сущности с группировкой по (book, chapter)
                    for entity_data in entities_raw:
                        entity_name = entity_data.get("name", "").strip()
                        if not entity_name:
                            continue

                        # Ключ = (имя, book, глава) для уникальности
                        entity_key = f"{entity_name}|{book_number}|{chapter}"

                        if entity_key not in entities_dict:
                            entities_dict[entity_key] = Entity(
                                name=entity_name,
                                type=entity_data.get("type", "UNKNOWN"),
                                book_number=book_number,
                                chapter=chapter,
                                descriptions_raw=[],
                                text_unit_ids=[],
                            )

                        # Накапливаем descriptions
                        description = entity_data.get("description", "").strip()
                        if description:
                            entities_dict[entity_key].descriptions_raw.append(description)
                        
                        entities_dict[entity_key].text_unit_ids.append(text_unit.id)
                        text_unit.entities.append(entity_key)  # Сохраняем ключ с главой

                    # Обрабатываем отношения с группировкой по (book, chapter)
                    for rel_data in relationships_raw:
                        source = rel_data.get("source", "").strip()
                        target = rel_data.get("target", "").strip()

                        if not source or not target:
                            continue

                        # Ключи с учетом book + chapter
                        source_key = f"{source}|{book_number}|{chapter}"
                        target_key = f"{target}|{book_number}|{chapter}"

                        # Проверяем что сущности существуют (с учетом book + chapter)
                        if source_key not in entities_dict or target_key not in entities_dict:
                            continue

                        # Генерируем MD5 ID (как в GraphRAG): hash(source|target|book|chapter)
                        rel_id = create_relationship_id(source, target, book_number, chapter)

                        if rel_id not in relationships_dict:
                            relationships_dict[rel_id] = Relationship(
                                source=source,
                                target=target,
                                book_number=book_number,
                                chapter=chapter,
                                descriptions_raw=[],
                                type=rel_data.get("type", "UNKNOWN"),
                                weight=rel_data.get("strength", 5) / 10.0,
                                text_unit_ids=[],
                            )

                        # Накапливаем descriptions
                        description = rel_data.get("description", "").strip()
                        if description:
                            relationships_dict[rel_id].descriptions_raw.append(description)
                        
                        relationships_dict[rel_id].text_unit_ids.append(text_unit.id)
                        text_unit.relationships.append(rel_id)

                except json.JSONDecodeError as e:
                    logger.error(
                        f"Ошибка парсинга JSON для {text_unit.id}: {e}\n"
                        f"Ответ: {response[:500]}"
                    )
                except Exception as e:
                    logger.error(f"Ошибка обработки {text_unit.id}: {e}")

            # Сохраняем checkpoint после обработки батча
            if checkpoint_dir:
                self._save_graph_checkpoint(
                    checkpoint_dir,
                    entities_dict,
                    relationships_dict,
                    batch_end,
                )
                logger.info(
                    f"✓ Checkpoint сохранен: {batch_end}/{total} чанков, "
                    f"{len(entities_dict)} entities, {len(relationships_dict)} relationships"
                )

        logger.info(
            f"Асинхронное извлечение завершено: {len(entities_dict)} сущностей, "
            f"{len(relationships_dict)} отношений"
        )

        # Суммаризируем descriptions_raw в финальное description
        logger.info("Суммаризация descriptions для entities и relationships...")
        for entity in entities_dict.values():
            if entity.descriptions_raw:
                # Простое объединение (TODO: добавить LLM суммаризацию)
                entity.description = " | ".join(set(entity.descriptions_raw))  # Убираем дубликаты
            else:
                entity.description = "Нет описания"
        
        for rel in relationships_dict.values():
            if rel.descriptions_raw:
                # Простое объединение (TODO: добавить LLM суммаризацию)
                rel.description = " | ".join(set(rel.descriptions_raw))  # Убираем дубликаты
            else:
                rel.description = "Нет описания"
        
        logger.info("✓ Суммаризация завершена")

        return entities_dict, relationships_dict

    def _save_graph_checkpoint(
        self,
        checkpoint_dir: Path,
        entities_dict: dict[str, Entity],
        relationships_dict: dict[str, Relationship],
        processed_count: int,
    ) -> None:
        """
        Сохраняет checkpoint процесса извлечения графа (Parquet).
        
        Сохраняет entities и relationships в РАЗНЫЕ файлы для:
        - Быстрой загрузки (меньше размер файлов)
        - Удобства отладки
        - Возможности загружать только нужное
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем processed_count в metadata
        metadata_file = checkpoint_dir / "step2_checkpoint_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump({"processed_count": processed_count}, f)
        
        # Используем ParquetStorage для сохранения
        temp_storage = ParquetStorage(checkpoint_dir.parent)
        temp_storage.data_path = checkpoint_dir
        
        # Сохраняем entities
        temp_storage.save_entities(entities_dict)
        (checkpoint_dir / "entities.parquet").rename(
            checkpoint_dir / "step2_checkpoint_entities.parquet"
        )
        
        # Сохраняем relationships
        temp_storage.save_relationships(relationships_dict)
        (checkpoint_dir / "relationships.parquet").rename(
            checkpoint_dir / "step2_checkpoint_relationships.parquet"
        )

    def _load_graph_checkpoint(
        self,
        checkpoint_dir: Path,
        text_units: list[TextUnit],
    ) -> tuple[dict[str, Entity], dict[str, Relationship], int]:
        """
        Загружает checkpoint процесса извлечения графа (Parquet).
        
        Загружает entities и relationships из РАЗНЫХ файлов.
        """
        # Загружаем processed_count из metadata
        metadata_file = checkpoint_dir / "step2_checkpoint_metadata.json"
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        processed_count = metadata["processed_count"]
        
        # Используем ParquetStorage для загрузки
        temp_storage = ParquetStorage(checkpoint_dir.parent)
        temp_storage.data_path = checkpoint_dir
        
        # Временно переименовываем для загрузки
        entities_checkpoint = checkpoint_dir / "step2_checkpoint_entities.parquet"
        entities_temp = checkpoint_dir / "entities.parquet"
        entities_checkpoint.rename(entities_temp)
        
        try:
            entities_loaded = temp_storage.load_entities()
        finally:
            entities_temp.rename(entities_checkpoint)
        
        # Загружаем relationships
        relationships_checkpoint = checkpoint_dir / "step2_checkpoint_relationships.parquet"
        relationships_temp = checkpoint_dir / "relationships.parquet"
        relationships_checkpoint.rename(relationships_temp)
        
        try:
            relationships_loaded = temp_storage.load_relationships()
        finally:
            relationships_temp.rename(relationships_checkpoint)
        
        # Ключи уже правильные после загрузки из ParquetStorage
        entities_dict = entities_loaded
        relationships_dict = relationships_loaded
        
        # Восстанавливаем связи в text_units
        for i in range(processed_count):
            if i < len(text_units):
                text_unit = text_units[i]
                # Восстанавливаем entities для этого text_unit
                for entity_name, entity in entities_dict.items():
                    if text_unit.id in entity.text_unit_ids:
                        if entity_name not in text_unit.entities:
                            text_unit.entities.append(entity_name)
                # Восстанавливаем relationships для этого text_unit
                for rel_id, rel in relationships_dict.items():
                    if text_unit.id in rel.text_unit_ids:
                        if rel_id not in text_unit.relationships:
                            text_unit.relationships.append(rel_id)
        
        return entities_dict, relationships_dict, processed_count

