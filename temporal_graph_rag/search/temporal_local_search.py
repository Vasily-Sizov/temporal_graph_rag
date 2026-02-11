"""Temporal Local Search - поиск с учетом временного контекста (Entity-centric)."""

import logging
import math
import re
from typing import Any

from transformers import AutoTokenizer

from temporal_graph_rag.models.data_models import (
    Community,
    Entity,
    Relationship,
    TextUnit,
)
from temporal_graph_rag.search.index_loader import IndexLoader
from temporal_graph_rag.settings import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SEARCH_MAX_CONTEXT_TOKENS,
)
from temporal_graph_rag.utils.api_client import APIClient, RerankerClient

logger = logging.getLogger(__name__)


class TemporalLocalSearch:
    """
    Local Search с поддержкой временных фильтров (Entity-centric как в GraphRAG).

    Использует поиск по entities вместо communities для более точных результатов.
    """

    def __init__(
        self,
        api_client: APIClient,
        index_loader: IndexLoader,
        text_units: list[TextUnit],
        entities_dict: dict[str, Entity],
        relationships_dict: dict[str, Relationship],
        communities: list[Community],
        top_k_entities: int = 10,
        top_k_communities: int = 5,
        top_k_relationships: int = 10,
        max_context_tokens: int = SEARCH_MAX_CONTEXT_TOKENS,
        text_unit_prop: float = 0.55,
        community_prop: float = 0.10,
        use_reranker: bool = True,
    ) -> None:
        """
        Инициализация Temporal Local Search.

        Args:
            api_client: Клиент для работы с API
            index_loader: Загрузчик индекса с FAISS
            text_units: Список текстовых единиц
            entities_dict: Словарь всех сущностей
            relationships_dict: Словарь всех отношений
            communities: Список всех communities
            top_k_entities: Количество entities для поиска
            top_k_communities: Количество communities для контекста
            top_k_relationships: Максимальное количество relationships на entity
            max_context_tokens: Максимальный размер контекста в токенах
            text_unit_prop: Доля токенов для text units (default 0.55 = 55%)
            community_prop: Доля токенов для communities (default 0.10 = 10%)
            use_reranker: Использовать ли reranker для переранжирования entities
        """
        self.api_client = api_client
        self.index_loader = index_loader
        self.text_units = text_units
        self.text_units_dict = {unit.id: unit for unit in text_units}
        self.entities_dict = entities_dict
        self.relationships_dict = relationships_dict
        self.communities = communities
        self.communities_dict = {comm.id: comm for comm in communities}
        self.top_k_entities = top_k_entities
        self.top_k_communities = top_k_communities
        self.top_k_relationships = top_k_relationships
        self.max_context_tokens = max_context_tokens
        self.text_unit_prop = text_unit_prop
        self.community_prop = community_prop
        self.use_reranker = use_reranker

        # Инициализируем reranker если нужно
        if self.use_reranker:
            try:
                self.reranker = RerankerClient()
                logger.info("Reranker клиент инициализирован")
            except Exception as e:
                logger.warning(
                    f"Не удалось инициализировать reranker: {e}. Reranking отключен."
                )
                self.use_reranker = False
                self.reranker = None
        else:
            self.reranker = None

        # Инициализируем tokenizer для точного подсчёта токенов (используем LLM tokenizer!)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL, trust_remote_code=True
            )
            logger.info(f"Загружен LLM tokenizer: {LLM_MODEL}")
        except Exception as e:
            logger.warning(
                f"Не удалось загрузить LLM tokenizer: {e}. Используем упрощённый подсчёт."
            )
            self.tokenizer = None

    def search(
        self,
        query: str,
        book_filter: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Выполняет поиск по запросу.

        Args:
            query: Поисковый запрос
            book_filter: Фильтр по номерам книг (опционально)

        Returns:
            Словарь с результатами поиска
        """
        logger.info(f"Поиск: {query}")
        if book_filter:
            logger.info(f"Фильтр по книгам: {book_filter}")

        # Шаг 1: Извлекаем временные подсказки из запроса
        if book_filter is None:
            book_filter = self._extract_book_hints(query)

        # Шаг 2: Находим релевантные entities (Entity-centric search)
        relevant_entities = self._find_relevant_entities(
            query=query,
            book_filter=book_filter,
        )

        logger.info(f"Найдено {len(relevant_entities)} релевантных entities")

        # Шаг 3: Находим связанные communities
        relevant_communities = self._find_communities_from_entities(
            entities=relevant_entities,
            book_filter=book_filter,
        )

        logger.info(f"Найдено {len(relevant_communities)} релевантных communities")

        # Шаг 4: Собираем relationships из найденных entities (для статистики)
        # Фильтруем по составному ключу (name, book_number, chapter)
        entity_keys = {
            f"{e.name}|{e.book_number}|{e.chapter}" for e in relevant_entities
        }
        relevant_relationships = [
            rel
            for rel in self.relationships_dict.values()
            if (
                f"{rel.source}|{rel.book_number}|{rel.chapter}" in entity_keys
                or f"{rel.target}|{rel.book_number}|{rel.chapter}" in entity_keys
            )
        ]
        logger.info(f"Найдено {len(relevant_relationships)} релевантных relationships")

        # Шаг 5: Строим контекст
        context = self._build_context(
            query=query,
            entities=relevant_entities,
            communities=relevant_communities,
        )

        # Шаг 6: Генерируем ответ
        answer, system_prompt, user_prompt = self._generate_answer(
            query=query, context=context
        )

        return {
            "query": query,
            "answer": answer,
            "entities_used": relevant_entities,
            "relationships_used": relevant_relationships,
            "communities_used": relevant_communities,
            "book_filter": book_filter,
            "context": context,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "text_units_dict": self.text_units_dict,
            "relationships_dict": self.relationships_dict,
        }

    def _extract_book_hints(self, query: str) -> list[int] | None:
        """
        Извлекает упоминания книг из запроса.

        Args:
            query: Поисковый запрос

        Returns:
            Список номеров книг или None
        """
        query_lower = query.lower()

        patterns = {
            r"в первой книге|философский камень|первая книга": [1],
            r"во второй книге|тайная комната|вторая книга": [2],
            r"в третьей книге|узник азкабана|третья книга": [3],
            r"в четвертой книге|кубок огня|четвертая книга": [4],
            r"в пятой книге|орден феникса|пятая книга": [5],
            r"в шестой книге|принц-полукровка|шестая книга": [6],
            r"в седьмой книге|дары смерти|седьмая книга": [7],
            r"в начале|сначала|впервые": [1, 2],
            r"в конце|финал|последн": [6, 7],
            r"как изменился|эволюция|развитие|на протяжении": list(range(1, 8)),
        }

        for pattern, books in patterns.items():
            if re.search(pattern, query_lower):
                logger.info(
                    f"Обнаружена временная подсказка: {pattern} -> книги {books}"
                )
                return books

        return None

    def _find_relevant_entities(
        self,
        query: str,
        book_filter: list[int] | None,
    ) -> list[Entity]:
        """
        Находит релевантные entities для запроса (Entity-centric search).

        Args:
            query: Поисковый запрос
            book_filter: Фильтр по книгам

        Returns:
            Список релевантных entities
        """
        # Получаем эмбеддинг запроса
        try:
            query_embedding = self.api_client.embed_single(query)
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддинга запроса: {e}")
            # Возвращаем первые N entities без ранжирования
            entities_list = list(self.entities_dict.values())
            return entities_list[: self.top_k_entities]

        # Определяем oversample multiplier (× 3 для reranking, × 2 без)
        oversample_mult = 3 if self.use_reranker else 2

        # Поиск в FAISS индексе
        search_results = self.index_loader.search_entities(
            query_embedding, k=self.top_k_entities * oversample_mult
        )

        logger.info(
            f"FAISS поиск: {len(search_results)} entities (top-{self.top_k_entities} × {oversample_mult})"
        )

        # Фильтруем по книгам если нужно
        candidate_entities = []
        for entity_name, similarity in search_results:
            entity = self.entities_dict.get(entity_name)
            if not entity:
                continue

            # Фильтр по книгам
            if book_filter and entity.book_number not in book_filter:
                continue

            candidate_entities.append((entity, similarity))

        logger.info(f"После фильтра по книгам: {len(candidate_entities)} entities")

        # Reranking если включен
        if self.use_reranker and self.reranker and candidate_entities:
            logger.info(f"Reranking {len(candidate_entities)} entities...")
            relevant_entities = self._rerank_entities(query, candidate_entities)
        else:
            # Без reranking берём первые top_k
            relevant_entities = [
                entity for entity, _ in candidate_entities[: self.top_k_entities]
            ]

        logger.info(f"Финально отобрано: {len(relevant_entities)} entities")
        return relevant_entities

    def _rerank_entities(
        self,
        query: str,
        candidate_entities: list[tuple[Entity, float]],
    ) -> list[Entity]:
        """
        Переранжирует entities через reranker.

        Args:
            query: Поисковый запрос
            candidate_entities: Список кортежей (Entity, faiss_similarity)

        Returns:
            Список переранжированных entities (top_k)
        """
        if not candidate_entities:
            return []

        # Формируем документы для reranker: "name: description"
        documents = []
        entities_list = []

        for entity, _ in candidate_entities:
            # Текст для reranking: имя + описание
            doc_text = f"{entity.name}: {entity.description}"
            documents.append(doc_text)
            entities_list.append(entity)

        logger.info(f"Отправка {len(documents)} entities в reranker...")

        try:
            # Переранжируем
            rerank_results = self.reranker.rerank(
                query=query,
                documents=documents,
                top_n=None,  # Переранжируем все
            )

            # Сортируем по rerank score и берём top_k
            reranked_entities = []
            for rerank_item in rerank_results[: self.top_k_entities]:
                doc_idx = rerank_item.get("index", -1)
                rerank_score = rerank_item.get("score", 0.0)

                if 0 <= doc_idx < len(entities_list):
                    entity = entities_list[doc_idx]
                    reranked_entities.append(entity)
                    logger.debug(
                        f"  Reranked: {entity.name} (score: {rerank_score:.4f})"
                    )

            logger.info(
                f"Reranking завершен: отобрано {len(reranked_entities)} entities"
            )
            return reranked_entities

        except Exception as e:
            logger.error(f"Ошибка reranking entities: {e}. Используем FAISS ranking.")
            # Fallback: возвращаем первые top_k без reranking
            return [entity for entity, _ in candidate_entities[: self.top_k_entities]]

    def _find_communities_from_entities(
        self,
        entities: list[Entity],
        book_filter: list[int] | None,
    ) -> list[Community]:
        """
        Находит communities связанные с найденными entities.

        Args:
            entities: Список entities
            book_filter: Фильтр по книгам

        Returns:
            Список релевантных communities
        """
        # Собираем все communities, которые содержат найденные entities
        # Теперь entity уже сгруппированы по (name, book, chapter),
        # поэтому ищем communities, содержащие эти ключи
        entity_keys = {f"{e.name}|{e.book_number}|{e.chapter}" for e in entities}

        relevant_communities = []
        for community in self.communities:
            # Применяем фильтр по книгам
            if book_filter and community.book_number not in book_filter:
                continue

            # Проверяем пересечение entities
            comm_entities = set(community.entities)
            overlap_count = len(entity_keys.intersection(comm_entities))

            if overlap_count > 0:
                relevant_communities.append((community, overlap_count))

        # Сортируем по количеству совпадающих entities (по убыванию)
        relevant_communities.sort(key=lambda x: x[1], reverse=True)

        # Возвращаем top-K
        return [comm for comm, _ in relevant_communities[: self.top_k_communities]]

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """
        Вычисляет cosine similarity между двумя векторами.

        Args:
            vec1: Первый вектор
            vec2: Второй вектор

        Returns:
            Cosine similarity (0.0-1.0)
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _build_context(
        self,
        query: str,
        entities: list[Entity],
        communities: list[Community],
    ) -> str:
        """
        Строит контекст для генерации ответа (как в GraphRAG от Microsoft).

        Использует token budget:
        - text_unit_prop (55%) для text units
        - community_prop (10%) для communities
        - остальное (35%) для entities + relationships

        Args:
            query: Поисковый запрос
            entities: Релевантные entities
            communities: Релевантные communities

        Returns:
            Текстовый контекст
        """
        context_parts = []

        # Подсчёт токенов (используем реальный tokenizer или упрощённый метод)
        def count_tokens(text: str) -> int:
            if self.tokenizer:
                try:
                    return len(self.tokenizer.encode(text, add_special_tokens=False))
                except Exception:
                    pass
            # Fallback: упрощённый подсчёт
            return len(text) // 4

        # Бюджет токенов для каждой секции
        community_tokens = int(self.max_context_tokens * self.community_prop)
        text_unit_tokens = int(self.max_context_tokens * self.text_unit_prop)
        local_tokens = int(
            self.max_context_tokens * (1 - self.community_prop - self.text_unit_prop)
        )

        # Секция 1: Community Reports (25% токенов)
        if communities:
            section_text = ["# Community Reports", ""]
            current_tokens = count_tokens("\n".join(section_text))

            for community in communities:
                comm_text = f"## {community.title}\n{community.summary}\n"
                comm_tokens = count_tokens(comm_text)

                if current_tokens + comm_tokens > community_tokens:
                    break

                section_text.append(f"## {community.title}")
                section_text.append(community.summary)
                section_text.append("")
                current_tokens += comm_tokens

            if len(section_text) > 2:  # Есть хоть одна community
                context_parts.extend(section_text)

        # Секция 2: Entities + Relationships (45% токенов)
        # Группируем по entities: Entity → её Relationships → следующая Entity
        local_section = []
        current_tokens = 0

        if entities:
            local_section.append("# Entities")
            local_section.append("")
            current_tokens = count_tokens("# Entities\n\n")

            # Собираем все relationships один раз
            # Создаем множество составных ключей для быстрой проверки
            entity_keys = {
                f"{e.name}|{e.book_number}|{e.chapter}" for e in entities
            }
            all_relationships = list(self.relationships_dict.values())

            # Постепенно добавляем entity + её relationships
            for entity in entities:
                # Добавляем entity
                entity_text = (
                    f"**{entity.name}** ({entity.type})\n- {entity.description}\n\n"
                )
                entity_tokens = count_tokens(entity_text)

                if current_tokens + entity_tokens > local_tokens:
                    break

                local_section.append(f"**{entity.name}** ({entity.type})")
                local_section.append(f"- {entity.description}")
                local_section.append("")
                current_tokens += entity_tokens

                # Текущий entity key
                current_entity_key = f"{entity.name}|{entity.book_number}|{entity.chapter}"

                # Находим relationships для этой entity по составному ключу
                # Отбираем relationships где source ИЛИ target совпадает по (name, book, chapter)
                entity_relationships = [
                    rel
                    for rel in all_relationships
                    if (
                        f"{rel.source}|{rel.book_number}|{rel.chapter}" == current_entity_key
                        or f"{rel.target}|{rel.book_number}|{rel.chapter}" == current_entity_key
                    )
                ]

                # In-network: с другими найденными entities (приоритет)
                entity_in_rels = [
                    rel
                    for rel in entity_relationships
                    if (
                        f"{rel.source}|{rel.book_number}|{rel.chapter}" in entity_keys
                        and f"{rel.target}|{rel.book_number}|{rel.chapter}" in entity_keys
                    )
                ]

                # Out-network: с entities вне списка
                entity_out_rels = [
                    rel
                    for rel in entity_relationships
                    if not (
                        f"{rel.source}|{rel.book_number}|{rel.chapter}" in entity_keys
                        and f"{rel.target}|{rel.book_number}|{rel.chapter}" in entity_keys
                    )
                ]

                # Сортируем внутри группы по временной позиции (relative_position)
                def get_temporal_position(rel: Relationship) -> float:
                    """
                    Получает временную позицию relationship по первому text_unit.
                    Используем relative_position из TextUnit для точной сортировки.
                    """
                    if not rel.text_unit_ids:
                        return 999.0  # В конец если нет text_units

                    # Берём первый text_unit_id
                    first_unit_id = rel.text_unit_ids[0]
                    text_unit = self.text_units_dict.get(first_unit_id)

                    if text_unit and hasattr(text_unit, "relative_position"):
                        return text_unit.relative_position

                    # Fallback: парсим номер из ID (например "book1_chunk_285" → 285)
                    try:
                        if "_chunk_" in first_unit_id:
                            chunk_num = int(first_unit_id.split("_chunk_")[1])
                            return float(chunk_num)
                    except (ValueError, IndexError):
                        pass

                    return 999.0

                entity_in_rels.sort(key=get_temporal_position)
                entity_out_rels.sort(key=get_temporal_position)

                # Объединяем: in-network + out-network
                entity_rels = entity_in_rels + entity_out_rels

                # Добавляем relationships для этой entity (ограничиваем до 10 на entity)
                if entity_rels:
                    # Простой заголовок без главы для LLM
                    local_section.append(f"### Relationships для {entity.name}:")
                    local_section.append("")

                    rels_header_tokens = count_tokens(
                        f"### Relationships для {entity.name}:\n\n"
                    )
                    current_tokens += rels_header_tokens

                    rel_count = 0
                    for rel in entity_rels:
                        if rel_count >= self.top_k_relationships:
                            break

                        rel_text = f"- **{rel.source}** → **{rel.target}**: {rel.description}\n"
                        rel_tokens = count_tokens(rel_text)

                        if current_tokens + rel_tokens > local_tokens:
                            break

                        local_section.append(
                            f"- **{rel.source}** → **{rel.target}**: {rel.description}"
                        )
                        current_tokens += rel_tokens
                        rel_count += 1

                    local_section.append("")

        context_parts.extend(local_section)

        # Секция 3: Sources / Text Units (55% токенов!)
        # Ранжируем text_units через reranker для семантической релевантности
        text_unit_info = []
        text_unit_ids_seen = set()

        # ЭТАП 1: Reranking text_units для каждой entity
        logger.info("Начало reranking text_units для всех entities")
        
        for entity_idx, entity in enumerate(entities):
            # Формируем контекст для reranker: query + entity info
            entity_context = f"{entity.name}: {entity.description[:200]}"
            query_with_context = f"{query}. Контекст: {entity_context}"
            
            # Собираем text_units ТОЛЬКО этой entity
            entity_text_units = []
            documents = []
            
            for text_unit_id in entity.text_unit_ids:
                if text_unit_id in text_unit_ids_seen:
                    continue
                
                text_unit = self.text_units_dict.get(text_unit_id)
                if not text_unit:
                    continue
                
                entity_text_units.append(text_unit)
                # Ограничиваем текст для reranker (макс 1000 символов)
                documents.append(text_unit.text[:1000])
                text_unit_ids_seen.add(text_unit_id)
            
            if not documents:
                continue
            
            # Отправляем в reranker текст_units этой entity
            try:
                rerank_results = self.reranker.rerank(
                    query=query_with_context,
                    documents=documents,
                    top_n=None  # Получаем все с scores
                )
                
                # Добавляем в общий список с rerank scores
                for result in rerank_results:
                    text_unit = entity_text_units[result['index']]
                    text_unit_info.append(
                        (
                            text_unit,
                            entity_idx,  # Порядок entity (для финальной сортировки)
                            result['score'],  # Rerank score для отбора
                            text_unit.temporal_position.relative_position,  # Для хронологии
                        )
                    )
                
                logger.info(
                    f"Entity #{entity_idx+1} ({entity.name}, {entity.chapter}): "
                    f"reranked {len(rerank_results)} text_units"
                )
            except Exception as e:
                logger.warning(
                    f"Ошибка reranking для entity {entity.name}: {e}. "
                    f"Пропускаем text_units этой entity."
                )
                continue

        # ЭТАП 2: Глобальная сортировка по rerank score (семантическая релевантность!)
        text_unit_info.sort(key=lambda x: -x[2])  # По убыванию score
        
        logger.info(
            f"Всего text_units после reranking: {len(text_unit_info)}, "
            f"отсортировано по rerank score"
        )

        # ЭТАП 3: Отбор по token budget (55% бюджета)
        selected_text_units = []
        if text_unit_info:
            temp_tokens = count_tokens("# Sources\n\n")

            for text_unit, entity_idx, rerank_score, rel_pos in text_unit_info:
                # Ограничиваем длину text_unit для безопасности (макс 2000 символов)
                text_content = text_unit.text[:2000]
                if len(text_unit.text) > 2000:
                    text_content += "..."

                source_text = f"**Source {text_unit.id}**:\n{text_content}\n\n"
                source_tokens = count_tokens(source_text)

                if temp_tokens + source_tokens > text_unit_tokens:
                    break

                selected_text_units.append(
                    (text_unit, entity_idx, rel_pos, text_content)
                )
                temp_tokens += source_tokens

        # ЭТАП 4: Финальная сортировка для вывода по relative_position (хронология)
        # Сортируем ТОЛЬКО по relative_position для временной последовательности событий
        selected_text_units.sort(key=lambda x: x[2])

        # ЭТАП 4: Добавляем в контекст в хронологическом порядке
        if selected_text_units:
            sources_section = ["# Sources", ""]

            for text_unit, entity_idx, rel_pos, text_content in selected_text_units:
                sources_section.append(f"**Source {text_unit.id}**:")
                sources_section.append(text_content)
                sources_section.append("")

            logger.info(f"Добавлено {len(selected_text_units)} text_units в контекст")

            if len(sources_section) > 2:  # Есть хоть один source
                context_parts.extend(sources_section)

        # Логируем общий размер контекста
        final_context = "\n".join(context_parts)
        total_tokens = count_tokens(final_context)
        logger.info(
            f"Общий размер контекста: ~{total_tokens} токенов (лимит: {self.max_context_tokens})"
        )

        if total_tokens > self.max_context_tokens:
            logger.warning(
                f"Контекст превышает лимит на {total_tokens - self.max_context_tokens} токенов!"
            )

        return final_context

    def _generate_answer(self, query: str, context: str) -> tuple[str, str, str]:
        """
        Генерирует ответ на основе контекста.

        Args:
            query: Поисковый запрос
            context: Контекстная информация

        Returns:
            Кортеж (answer, system_prompt, user_prompt)
        """
        system_prompt = """Ты — помощник, отвечающий на вопросы по книгам о Гарри Поттере.

Используй ТОЛЬКО информацию из предоставленного контекста для ответа.
Если в контексте нет информации для ответа, скажи об этом.

ВАЖНО:
- НЕ придумывай цитаты или ссылки на Source
- Если ссылаешься на Source, используй ТОЧНЫЙ текст из контекста
- НЕ добавляй информацию, которой нет в контексте

Структура ответа:
1. Прямой ответ на вопрос
2. Подтверждающие детали из контекста
3. Ссылки на конкретные книги/события если релевантно

Будь точным и конкретным."""

        prompt = f"""КОНТЕКСТ:
{context}

ВОПРОС:
{query}

ОТВЕТ:"""

        try:
            answer = self.api_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            return answer, system_prompt, prompt
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return f"Ошибка генерации ответа: {e}", system_prompt, prompt
