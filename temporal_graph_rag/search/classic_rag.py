"""Классический RAG с FAISS, Embedder, Reranker и LLM."""

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from temporal_graph_rag.prompts.classic_rag_prompt import CLASSIC_RAG_SYSTEM_PROMPT
from temporal_graph_rag.settings import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    SEARCH_MAX_CONTEXT_TOKENS,
)
from temporal_graph_rag.utils.api_client import APIClient, RerankerClient

logger = logging.getLogger(__name__)


class ClassicRAG:
    """
    Классический RAG с векторным поиском.

    Использует:
    - FAISS для векторного поиска
    - Embedder для получения векторов запроса
    - Reranker для переранжирования результатов (опционально)
    - LLM для генерации ответов
    """

    def __init__(
        self,
        index_dir: Path | str,
        api_client: APIClient,
        top_k_search: int = 50,
        top_k_rerank: int = 5,
        max_context_tokens: int = SEARCH_MAX_CONTEXT_TOKENS,
        use_reranker: bool = True,
    ) -> None:
        """
        Инициализация классического RAG.

        Args:
            index_dir: Директория с индексом (должна содержать text_units.faiss и text_units.json)
            api_client: Клиент для работы с API
            top_k_search: Количество документов для поиска в FAISS
            top_k_rerank: Количество документов после reranking
            max_context_tokens: Максимальное количество токенов в контексте
            use_reranker: Использовать ли reranker
        """
        self.api_client = api_client
        self.top_k_search = top_k_search
        self.top_k_rerank = top_k_rerank
        self.max_context_tokens = max_context_tokens
        self.use_reranker = use_reranker

        # Конвертируем в Path если строка
        index_dir = Path(index_dir)

        # Загружаем FAISS индекс (ожидаем структуру temporal_graph_rag)
        faiss_path = index_dir / "vectors" / "text_units.faiss"
        logger.info(f"Загрузка FAISS индекса из: {faiss_path}")
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS индекс не найден: {faiss_path}")
        self.index = faiss.read_index(str(faiss_path))
        logger.info(f"FAISS индекс загружен: {self.index.ntotal} векторов")

        # Загружаем метаданные из parquet
        metadata_path = index_dir / "data" / "text_units.parquet"
        logger.info(f"Загрузка метаданных из: {metadata_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Метаданные не найдены: {metadata_path}")
        
        # Читаем parquet и преобразуем в список словарей
        df = pd.read_parquet(metadata_path)
        self.metadata = []
        for idx, row in df.iterrows():
            # Парсим temporal_position
            temp_pos = json.loads(row["temporal_position"])
            
            chunk_dict = {
                "id": row["id"],
                "chunk_id": row["id"],
                "text": row["text"],
                "book_number": temp_pos["book_number"],
                "chapter": temp_pos.get("chapter", temp_pos.get("chunk_index", "?")),
                "book_title": temp_pos.get("book_title", ""),
                "chunk_index": temp_pos.get("chunk_index", 0),
                "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
            }
            self.metadata.append(chunk_dict)
        
        logger.info(f"Метаданные загружены: {len(self.metadata)} чанков")

        # Инициализируем reranker если нужен
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

        # Инициализируем tokenizer для подсчета токенов
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            logger.info(f"Tokenizer загружен для модели: {LLM_MODEL}")
        except Exception as e:
            logger.warning(f"Не удалось загрузить tokenizer: {e}. Используем упрощенный подсчет.")
            self.tokenizer = None

    def _count_tokens(self, text: str) -> int:
        """
        Подсчитывает количество токенов в тексте.

        Args:
            text: Текст для подсчета

        Returns:
            Количество токенов
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        # Fallback: упрощенный подсчет (1 токен ≈ 4 символа)
        return len(text) // 4

    def search(self, query: str) -> list[dict[str, Any]]:
        """
        Выполняет векторный поиск по запросу.

        Args:
            query: Поисковый запрос

        Returns:
            Список найденных чанков с оценками релевантности
        """
        logger.info(f"Векторный поиск для запроса: '{query[:50]}...'")

        # Получаем embedding запроса
        logger.debug("Получение embedding для запроса...")
        query_embedding = self.api_client.embed_single(query)
        query_vector = np.array([query_embedding], dtype=np.float32)

        # Нормализуем для cosine similarity
        faiss.normalize_L2(query_vector)

        # Поиск в FAISS
        logger.debug(f"Поиск топ-{self.top_k_search} документов в FAISS...")
        scores, indices = self.index.search(query_vector, self.top_k_search)

        # Извлекаем найденные чанки
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            # Проверяем валидность индекса
            if idx >= 0 and idx < len(self.metadata):
                chunk = self.metadata[idx]
                results.append(
                    {
                        "chunk": chunk,
                        "faiss_score": float(score),
                        "faiss_rank": i + 1,
                    }
                )
            else:
                logger.warning(
                    f"Некорректный индекс из FAISS: {idx} "
                    f"(всего метаданных: {len(self.metadata)})"
                )

        logger.info(f"Найдено {len(results)} документов")
        return results

    def rerank(
        self,
        query: str,
        search_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Переранжирует результаты поиска.

        Args:
            query: Поисковый запрос
            search_results: Результаты поиска из FAISS

        Returns:
            Переранжированные результаты
        """
        if not search_results:
            return []

        if not self.use_reranker or not self.reranker:
            logger.info("Reranker отключен, пропускаем переранжирование")
            return search_results[: self.top_k_rerank]

        logger.info(f"Реранжирование {len(search_results)} документов")

        # Извлекаем тексты для reranking
        documents = []
        chunk_ids = []
        for result in search_results:
            chunk = result["chunk"]
            chunk_id = chunk.get("chunk_id", chunk.get("id", "unknown"))
            text = chunk.get("text", "")

            if not text or not text.strip():
                logger.warning(f"Пропуск пустого чанка {chunk_id}")
                continue

            documents.append(text)
            chunk_ids.append(chunk_id)

        logger.debug(f"Отправка {len(documents)} документов в reranker")

        # Переранжируем все документы
        rerank_results = self.reranker.rerank(
            query=query,
            documents=documents,
            top_n=None,  # Переранжируем все
        )

        logger.debug(f"Reranker вернул {len(rerank_results)} результатов")

        # Объединяем результаты
        reranked = []
        for rerank_item in rerank_results:
            doc_idx = rerank_item.get("index", -1)
            rerank_score = rerank_item.get("score", 0.0)

            if 0 <= doc_idx < len(search_results):
                result = search_results[doc_idx].copy()
                result["rerank_score"] = rerank_score
                result["rerank_rank"] = len(reranked) + 1
                reranked.append(result)
            else:
                logger.warning(
                    f"Некорректный doc_idx={doc_idx} "
                    f"(всего документов: {len(documents)})"
                )

        logger.info(f"Переранжировано: {len(reranked)} документов")

        # Берем только топ K для LLM
        top_reranked = reranked[: self.top_k_rerank]
        logger.info(
            f"Взято для LLM: {len(top_reranked)} из {len(reranked)} переранжированных документов"
        )
        return top_reranked

    def generate_answer(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
    ) -> tuple[str, str, str]:
        """
        Генерирует ответ на основе запроса и контекста.

        Args:
            query: Поисковый запрос
            context_chunks: Чанки с контекстом

        Returns:
            Кортеж (answer, system_prompt, user_prompt)
        """
        logger.info("Генерация ответа через LLM")

        # Формируем контекст из чанков с учетом лимита токенов
        context_parts = []
        total_tokens = 0

        for i, result in enumerate(context_chunks, 1):
            chunk = result["chunk"]
            text = chunk.get("text", "")

            # Подсчитываем токены
            chunk_tokens = self._count_tokens(text)

            # Проверяем, не превысим ли лимит
            if total_tokens + chunk_tokens > self.max_context_tokens:
                logger.info(
                    f"Контекст ограничен: использовано {i - 1} из {len(context_chunks)} чанков "
                    f"({total_tokens} токенов из {self.max_context_tokens})"
                )
                break

            # Получаем метаданные чанка
            chunk_id = chunk.get("chunk_id", chunk.get("id", "unknown"))
            book_number = chunk.get("book_number", "?")
            chapter = chunk.get("chapter", "?")

            context_parts.append(
                f"[Документ {i}] (Книга {book_number}, Глава {chapter}, ID: {chunk_id})\n{text}"
            )
            total_tokens += chunk_tokens

        context = "\n\n".join(context_parts)

        logger.info(
            f"Контекст сформирован: {len(context_parts)} чанков, ~{total_tokens} токенов"
        )

        # Формируем промпты
        system_prompt = CLASSIC_RAG_SYSTEM_PROMPT

        user_prompt = f"""КОНТЕКСТ:
{context}

ВОПРОС:
{query}

ОТВЕТ:"""

        try:
            answer = self.api_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            return answer, system_prompt, user_prompt
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return f"Ошибка генерации ответа: {e}", system_prompt, user_prompt

    def query(self, query: str) -> dict[str, Any]:
        """
        Выполняет полный RAG запрос.

        Args:
            query: Поисковый запрос

        Returns:
            Словарь с результатами:
            - query: исходный запрос
            - answer: сгенерированный ответ
            - context_chunks: использованные чанки
            - search_results: все результаты поиска
            - system_prompt: использованный system prompt
            - user_prompt: использованный user prompt
        """
        logger.info(f"Classic RAG запрос: '{query}'")

        # 1. Векторный поиск
        search_results = self.search(query)

        # 2. Переранжирование (опционально)
        if self.use_reranker and search_results:
            context_chunks = self.rerank(query, search_results)
        else:
            context_chunks = search_results[: self.top_k_rerank]

        # 3. Генерация ответа
        answer, system_prompt, user_prompt = self.generate_answer(query, context_chunks)

        return {
            "query": query,
            "answer": answer,
            "context_chunks": context_chunks,
            "search_results": search_results,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

