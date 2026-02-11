"""Построение temporal communities с использованием sliding window."""

import asyncio
import json
import logging
from typing import Any

from temporal_graph_rag.models.data_models import (
    Community,
    Entity,
    Relationship,
    TextUnit,
)
from temporal_graph_rag.prompts import COMMUNITY_REPORT_PROMPT
from temporal_graph_rag.settings import (
    COMMUNITY_REPORT_MAX_TOKENS,
    COMMUNITY_REPORT_TEMPERATURE,
)
from temporal_graph_rag.utils.api_client import APIClient

logger = logging.getLogger(__name__)


class CommunityBuilder:
    """Строит temporal communities используя sliding window подход."""

    def __init__(
        self,
        api_client: APIClient,
        window_size: int = 20,
        overlap: int = 5,
    ) -> None:
        """
        Инициализация построителя сообществ.

        Args:
            api_client: Клиент для работы с API
            window_size: Размер окна в чанках
            overlap: Перекрытие между окнами в чанках
        """
        self.api_client = api_client
        self.window_size = window_size
        self.overlap = overlap

    def create_temporal_communities(
        self,
        text_units: list[TextUnit],
        entities_dict: dict[str, Entity],
        relationships_dict: dict[str, Relationship],
    ) -> list[Community]:
        """
        Создает temporal communities для списка текстовых единиц.

        Args:
            text_units: Список текстовых единиц (должны быть из одной книги)
            entities_dict: Словарь всех сущностей
            relationships_dict: Словарь всех отношений

        Returns:
            Список Community объектов
        """
        if not text_units:
            return []

        book_number = text_units[0].temporal_position.book_number
        book_title = text_units[0].temporal_position.book_title

        logger.info(
            f"Создание temporal communities для книги {book_number}: {book_title}"
        )
        logger.info(
            f"Параметры: window_size={self.window_size}, overlap={self.overlap}"
        )

        communities = []
        step = self.window_size - self.overlap

        for i in range(0, len(text_units), step):
            window_units = text_units[i : i + self.window_size]

            if not window_units:
                continue

            # Создаем сообщество для этого окна
            community = self._create_community_for_window(
                window_units=window_units,
                window_index=len(communities),
                book_number=book_number,
                book_title=book_title,
                entities_dict=entities_dict,
                relationships_dict=relationships_dict,
            )

            if community:
                communities.append(community)

        logger.info(f"Создано {len(communities)} temporal communities")

        # Temporal instances больше не используются - плоская структура (book+chapter)
        # self._create_temporal_instances() - удалено

        return communities

    def _create_community_for_window(
        self,
        window_units: list[TextUnit],
        window_index: int,
        book_number: int,
        book_title: str,
        entities_dict: dict[str, Entity],
        relationships_dict: dict[str, Relationship],
    ) -> Community | None:
        """
        Создает сообщество для одного временного окна.

        Args:
            window_units: Текстовые единицы в окне
            window_index: Индекс окна
            book_number: Номер книги
            book_title: Название книги
            entities_dict: Словарь всех сущностей
            relationships_dict: Словарь всех отношений

        Returns:
            Community объект или None при ошибке
        """
        # Собираем сущности и отношения из окна
        entities_in_window = set()
        relationships_in_window = set()
        text_unit_ids = []

        for unit in window_units:
            text_unit_ids.append(unit.id)
            entities_in_window.update(unit.entities)
            relationships_in_window.update(unit.relationships)

        if not entities_in_window:
            logger.warning(f"Нет сущностей в окне {window_index}, пропускаем")
            return None

        # Определяем временной диапазон
        start_chunk = window_units[0].temporal_position.chunk_index
        end_chunk = window_units[-1].temporal_position.chunk_index
        start_position = window_units[0].temporal_position.relative_position
        end_position = window_units[-1].temporal_position.relative_position

        # Генерируем отчет о сообществе
        title, summary, report = self._generate_community_report(
            entities_in_window=list(entities_in_window),
            relationships_in_window=list(relationships_in_window),
            entities_dict=entities_dict,
            relationships_dict=relationships_dict,
            book_title=book_title,
            book_number=book_number,
            start_chunk=start_chunk,
            end_chunk=end_chunk,
            start_position=start_position,
            end_position=end_position,
        )

        # Создаем сообщество
        community = Community(
            id=f"book{book_number}_window_{window_index}",
            book_number=book_number,
            temporal_range=(start_chunk, end_chunk),
            relative_position_range=(start_position, end_position),
            title=title,
            summary=summary,
            report=report,
            entities=list(entities_in_window),
            relationships=list(relationships_in_window),
            text_unit_ids=text_unit_ids,
        )

        logger.debug(
            f"Создано сообщество {community.id}: "
            f"{len(entities_in_window)} сущностей, "
            f"{len(relationships_in_window)} отношений"
        )

        return community

    def _generate_community_report(
        self,
        entities_in_window: list[str],
        relationships_in_window: list[str],
        entities_dict: dict[str, Entity],
        relationships_dict: dict[str, Relationship],
        book_title: str,
        book_number: int,
        start_chunk: int,
        end_chunk: int,
        start_position: float,
        end_position: float,
    ) -> tuple[str, str, str]:
        """
        Генерирует отчет о сообществе с помощью LLM.

        Returns:
            Кортеж (title, summary, report)
        """
        # ВАЖНО: Сортируем entities по book_number и chapter для хронологического порядка
        sorted_entities = []
        for entity_name in entities_in_window:
            entity = entities_dict.get(entity_name)
            if entity:
                sorted_entities.append(entity)
        
        # Сортируем по (book_number, chapter) для хронологического порядка
        sorted_entities.sort(key=lambda e: (e.book_number, e.chapter))
        
        # Форматируем сущности
        entities_text = []
        for entity in sorted_entities[:50]:  # Ограничиваем для контекста
            entities_text.append(
                f"- {entity.name} ({entity.type}): {entity.description}"
            )

        # ВАЖНО: Сортируем relationships по book_number и chapter для хронологического порядка
        sorted_relationships = []
        for rel_id in relationships_in_window:
            rel = relationships_dict.get(rel_id)
            if rel:
                sorted_relationships.append(rel)
        
        # Сортируем по (book_number, chapter) для хронологического порядка
        sorted_relationships.sort(key=lambda r: (r.book_number, r.chapter))
        
        # Форматируем отношения
        relationships_text = []
        for rel in sorted_relationships[:50]:  # Ограничиваем для контекста
            relationships_text.append(
                f"- {rel.source} → {rel.target}: {rel.description}"
            )

        # Формируем промпт
        prompt = COMMUNITY_REPORT_PROMPT.format(
            book_title=book_title,
            book_number=book_number,
            start_chunk=start_chunk,
            end_chunk=end_chunk,
            start_position=start_position,
            end_position=end_position,
            entities="\n".join(entities_text) if entities_text else "Нет данных",
            relationships="\n".join(relationships_text)
            if relationships_text
            else "Нет данных",
        )

        try:
            # Генерируем отчет
            response = self.api_client.generate(
                prompt=prompt,
                temperature=COMMUNITY_REPORT_TEMPERATURE,
                max_tokens=COMMUNITY_REPORT_MAX_TOKENS,
            )

            # Парсим JSON
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            data = json.loads(response)

            title = data.get("title", f"Окно {start_chunk}-{end_chunk}")
            summary = data.get("summary", "")
            report = data.get("report", "")

            return title, summary, report

        except Exception as e:
            logger.error(f"Ошибка генерации отчета: {e}")
            return (
                f"Окно {start_chunk}-{end_chunk}",
                f"События в чанках {start_chunk}-{end_chunk}",
                f"Сообщество содержит {len(entities_in_window)} сущностей и "
                f"{len(relationships_in_window)} отношений.",
            )

    def _create_temporal_instances(
        self,
        text_units: list[TextUnit],
        communities: list[Community],
        entities_dict: dict[str, Entity],
        relationships_dict: dict[str, Relationship],
    ) -> None:
        """
        Создает temporal instances для сущностей и отношений.
        
        Группирует по главам (используя headings из metadata) вместо communities.

        Args:
            text_units: Список текстовых единиц
            communities: Список сообществ
            entities_dict: Словарь сущностей (будет модифицирован)
            relationships_dict: Словарь отношений (будет модифицирован)
        """
        logger.info("Создание temporal instances для сущностей и отношений (группировка по главам)")

        # Группируем communities по главам используя headings из text_units
        chapters_data = self._group_by_chapters(text_units, communities, entities_dict, relationships_dict)
        
        logger.info(f"Найдено {len(chapters_data)} глав для группировки")

        # Создаем temporal instances для каждой главы
        for chapter_key, chapter_info in chapters_data.items():
            book_number = chapter_info["book_number"]
            chapter_heading = chapter_info["heading"]
            entities_in_chapter = chapter_info["entities"]
            relationships_in_chapter = chapter_info["relationships"]
            temporal_range = chapter_info["temporal_range"]
            
            # Создаем instances для сущностей в этой главе
            for entity_name in entities_in_chapter:
                entity = entities_dict.get(entity_name)
                if not entity:
                    continue

                # Локальное описание = глобальное описание (пока без суммаризации)
                # TODO: Добавить суммаризацию descriptions в пределах главы
                local_description = entity.description

                # Создаем temporal instance для главы
                instance = EntityTemporalInstance(
                    entity_name=entity_name,
                    book_number=book_number,
                    community_id=chapter_key,  # ID главы
                    temporal_range=temporal_range,
                    local_description=local_description,
                    local_relationships=[
                        rel_id
                        for rel_id in relationships_in_chapter
                        if relationships_dict.get(rel_id)
                        and (
                            relationships_dict[rel_id].source == entity_name
                            or relationships_dict[rel_id].target == entity_name
                        )
                    ],
                    importance=0.5,
                )

                entity.temporal_instances.append(instance)

            # Создаем instances для отношений в этой главе
            for rel_id in relationships_in_chapter:
                rel = relationships_dict.get(rel_id)
                if not rel:
                    continue

                instance = RelationshipTemporalInstance(
                    source=rel.source,
                    target=rel.target,
                    book_number=book_number,
                    community_id=chapter_key,
                    temporal_range=temporal_range,
                    local_description=rel.description,
                    strength=int(rel.weight * 10),
                )

                rel.temporal_instances.append(instance)

        logger.info(f"Temporal instances созданы для {len(chapters_data)} глав")

    def _group_by_chapters(
        self,
        text_units: list[TextUnit],
        communities: list[Community],
        entities_dict: dict[str, Entity],
        relationships_dict: dict[str, Relationship],
    ) -> dict[str, dict[str, Any]]:
        """
        Группирует сущности и отношения по главам используя headings из text_units.
        
        Args:
            text_units: Список текстовых единиц
            communities: Список communities
            entities_dict: Словарь сущностей
            relationships_dict: Словарь отношений
            
        Returns:
            Словарь: chapter_key -> {book_number, heading, entities, relationships, temporal_range}
        """
        # Создаем словарь text_unit_id -> headings для быстрого поиска
        text_unit_headings = {}
        for tu in text_units:
            headings = tu.metadata.get("headings", [])
            # Берем первый heading, игнорируя "Annotation"
            chapter_heading = None
            for h in headings:
                if h not in ["Annotation", "annotation"]:
                    chapter_heading = h
                    break
            if chapter_heading:
                text_unit_headings[tu.id] = chapter_heading
        
        chapters: dict[str, dict[str, Any]] = {}
        
        for community in communities:
            # Определяем к какой главе относится community
            # Берем heading первого text_unit в community
            chapter_heading = None
            for text_unit_id in community.text_unit_ids:
                if text_unit_id in text_unit_headings:
                    chapter_heading = text_unit_headings[text_unit_id]
                    break
            
            if not chapter_heading:
                # Если не нашли heading, используем community как "главу"
                chapter_heading = community.title
            
            chapter_key = f"book{community.book_number}_{chapter_heading}"
            
            if chapter_key not in chapters:
                chapters[chapter_key] = {
                    "book_number": community.book_number,
                    "heading": chapter_heading,
                    "entities": set(),
                    "relationships": set(),
                    "temporal_range": community.temporal_range,
                }
            else:
                # Расширяем temporal_range если нужно
                existing_range = chapters[chapter_key]["temporal_range"]
                new_range = (
                    min(existing_range[0], community.temporal_range[0]),
                    max(existing_range[1], community.temporal_range[1]),
                )
                chapters[chapter_key]["temporal_range"] = new_range
            
            # Добавляем сущности и отношения
            chapters[chapter_key]["entities"].update(community.entities)
            chapters[chapter_key]["relationships"].update(community.relationships)
        
        return chapters

    async def create_temporal_communities_async(
        self,
        text_units: list[TextUnit],
        entities_dict: dict[str, Entity],
        relationships_dict: dict[str, Relationship],
    ) -> list[Community]:
        """
        Асинхронно создает temporal communities для списка текстовых единиц.
        
        Использует параллельные запросы к LLM для генерации community reports.

        Args:
            text_units: Список текстовых единиц (должны быть из одной книги)
            entities_dict: Словарь всех сущностей
            relationships_dict: Словарь всех отношений

        Returns:
            Список Community объектов
        """
        if not text_units:
            return []

        book_number = text_units[0].temporal_position.book_number
        book_title = text_units[0].temporal_position.book_title

        logger.info(
            f"АСИНХРОННОЕ создание temporal communities для книги {book_number}: {book_title}"
        )
        logger.info(
            f"Параметры: window_size={self.window_size}, overlap={self.overlap}"
        )

        # Шаг 1: Собираем все окна и их метаданные
        communities = []
        prompts = []
        window_metadata = []
        
        step = self.window_size - self.overlap

        for i in range(0, len(text_units), step):
            window_units = text_units[i : i + self.window_size]
            if not window_units:
                continue

            # Собираем сущности и отношения из окна
            entities_in_window = set()
            relationships_in_window = set()
            text_unit_ids = []

            for unit in window_units:
                text_unit_ids.append(unit.id)
                entities_in_window.update(unit.entities)
                relationships_in_window.update(unit.relationships)

            if not entities_in_window:
                logger.warning(f"Нет сущностей в окне {len(communities)}, пропускаем")
                continue

            # Определяем временной диапазон
            start_chunk = window_units[0].temporal_position.chunk_index
            end_chunk = window_units[-1].temporal_position.chunk_index
            start_position = window_units[0].temporal_position.relative_position
            end_position = window_units[-1].temporal_position.relative_position

            # ВАЖНО: Сортируем entities и relationships по хронологии
            sorted_entities = []
            for entity_name in entities_in_window:
                entity = entities_dict.get(entity_name)
                if entity:
                    sorted_entities.append(entity)
            sorted_entities.sort(key=lambda e: (e.book_number, e.chapter))
            
            sorted_relationships = []
            for rel_id in relationships_in_window:
                rel = relationships_dict.get(rel_id)
                if rel:
                    sorted_relationships.append(rel)
            sorted_relationships.sort(key=lambda r: (r.book_number, r.chapter))
            
            # Форматируем сущности и отношения для промпта
            entities_text = []
            for entity in sorted_entities[:50]:
                entities_text.append(
                    f"- {entity.name} ({entity.type}): {entity.description}"
                )

            relationships_text = []
            for rel in sorted_relationships[:50]:
                relationships_text.append(
                    f"- {rel.source} → {rel.target}: {rel.description}"
                )

            # Формируем промпт
            prompt = COMMUNITY_REPORT_PROMPT.format(
                book_title=book_title,
                book_number=book_number,
                start_chunk=start_chunk,
                end_chunk=end_chunk,
                start_position=start_position,
                end_position=end_position,
                entities="\n".join(entities_text) if entities_text else "Нет данных",
                relationships="\n".join(relationships_text)
                if relationships_text
                else "Нет данных",
            )

            prompts.append(prompt)
            window_metadata.append({
                "window_index": len(communities),
                "book_number": book_number,
                "temporal_range": (start_chunk, end_chunk),
                "relative_position_range": (start_position, end_position),
                "entities": list(entities_in_window),
                "relationships": list(relationships_in_window),
                "text_unit_ids": text_unit_ids,
            })

        # Шаг 2: Параллельная генерация отчетов
        logger.info(f"Генерация {len(prompts)} community reports...")
        responses = await self.api_client.generate_batch_async(
            prompts=prompts,
            temperature=COMMUNITY_REPORT_TEMPERATURE,
            max_tokens=COMMUNITY_REPORT_MAX_TOKENS,
        )

        # Шаг 3: Создаем Community объекты
        for metadata, response in zip(window_metadata, responses):
            try:
                # Парсим JSON
                response = response.strip()
                if response.startswith("```"):
                    lines = response.split("\n")
                    response = "\n".join(lines[1:-1])

                data = json.loads(response)
                title = data.get("title", f"Окно {metadata['temporal_range'][0]}-{metadata['temporal_range'][1]}")
                summary = data.get("summary", "")
                report = data.get("report", "")

            except Exception as e:
                logger.error(f"Ошибка парсинга отчета для окна {metadata['window_index']}: {e}")
                title = f"Окно {metadata['temporal_range'][0]}-{metadata['temporal_range'][1]}"
                summary = f"События в чанках {metadata['temporal_range'][0]}-{metadata['temporal_range'][1]}"
                report = f"Сообщество содержит {len(metadata['entities'])} сущностей."

            # Создаем сообщество
            community = Community(
                id=f"book{metadata['book_number']}_window_{metadata['window_index']}",
                book_number=metadata["book_number"],
                temporal_range=metadata["temporal_range"],
                relative_position_range=metadata["relative_position_range"],
                title=title,
                summary=summary,
                report=report,
                entities=metadata["entities"],
                relationships=metadata["relationships"],
                text_unit_ids=metadata["text_unit_ids"],
            )

            communities.append(community)

        logger.info(f"Создано {len(communities)} temporal communities")

        # Temporal instances больше не используются - плоская структура (book+chapter)
        # self._create_temporal_instances() - удалено

        return communities

