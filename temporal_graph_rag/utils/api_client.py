"""Клиент для работы с API LLM и Embedder."""

import asyncio
import logging
import random
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import requests
from transformers import AutoTokenizer

from temporal_graph_rag.settings import (
    EMBEDDER_BATCH_SIZE,
    EMBEDDER_CONCURRENT_REQUESTS,
    EMBEDDER_ENDPOINT,
    EMBEDDER_MAX_RETRIES,
    EMBEDDER_MAX_TOKENS,
    EMBEDDER_MODEL,
    EMBEDDER_REQUEST_TIMEOUT,
    EMBEDDER_RETRY_BASE_DELAY_SECONDS,
    EMBEDDER_RETRY_JITTER,
    EMBEDDER_RETRY_MAX_DELAY_SECONDS,
    LLM_CONCURRENT_REQUESTS,
    LLM_ENDPOINT,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_REQUEST_TIMEOUT,
    LLM_RETRY_BASE_DELAY_SECONDS,
    LLM_RETRY_JITTER,
    LLM_RETRY_MAX_DELAY_SECONDS,
    LLM_TEMPERATURE,
    RERANKER_ENDPOINT,
    RERANKER_MODEL,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


# Глобальные семафоры для контроля параллелизма
# Создаются при первом использовании асинхронных методов
_llm_semaphore: asyncio.Semaphore | None = None
_embedder_semaphore: asyncio.Semaphore | None = None


def _get_llm_semaphore() -> asyncio.Semaphore:
    """Получает или создает семафор для LLM запросов."""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(LLM_CONCURRENT_REQUESTS)
        logger.info(f"Создан LLM семафор: max {LLM_CONCURRENT_REQUESTS} параллельных запросов")
    return _llm_semaphore


def _get_embedder_semaphore() -> asyncio.Semaphore:
    """Получает или создает семафор для Embedder запросов."""
    global _embedder_semaphore
    if _embedder_semaphore is None:
        _embedder_semaphore = asyncio.Semaphore(EMBEDDER_CONCURRENT_REQUESTS)
        logger.info(f"Создан Embedder семафор: max {EMBEDDER_CONCURRENT_REQUESTS} параллельных запросов")
    return _embedder_semaphore


class APIClient:
    """
    Универсальный клиент для работы с LLM и Embedder API с retry механизмом.

    ВАЖНО ПО ПАРАЛЛЕЛИЗМУ (соответствует --max-num-seqs в docker-compose.yml):
    - LLM: может обрабатывать до 8 запросов параллельно (--max-num-seqs 8)
    - Embedder: может обрабатывать только 1 запрос за раз (--max-num-seqs 1)

    Текущая реализация использует последовательные запросы. Для параллелизма
    LLM запросов можно использовать asyncio или ThreadPoolExecutor.

    Для Embedder критически важно отправлять по 1 тексту за раз (или батч размером 1),
    так как сервер настроен на обработку только одного запроса одновременно.
    """
    
    _tokenizer = None  # Кэш токенайзера для embedder

    def __init__(
        self,
        llm_endpoint: str = LLM_ENDPOINT,
        embedder_endpoint: str = EMBEDDER_ENDPOINT,
        llm_model: str = LLM_MODEL,
        embedder_model: str = EMBEDDER_MODEL,
        llm_max_retries: int = LLM_MAX_RETRIES,
        llm_timeout: int = LLM_REQUEST_TIMEOUT,
        embedder_max_retries: int = EMBEDDER_MAX_RETRIES,
        embedder_timeout: int = EMBEDDER_REQUEST_TIMEOUT,
        embedder_batch_size: int = EMBEDDER_BATCH_SIZE,
        embedder_max_tokens: int = EMBEDDER_MAX_TOKENS,
    ) -> None:
        """
        Инициализация API клиента.

        Args:
            llm_endpoint: URL для LLM API
            embedder_endpoint: URL для Embedder API
            llm_model: Название LLM модели
            embedder_model: Название Embedder модели
            llm_max_retries: Максимальное количество повторов для LLM
            llm_timeout: Таймаут запроса к LLM в секундах
            embedder_max_retries: Максимальное количество повторов для Embedder
            embedder_timeout: Таймаут запроса к Embedder в секундах
            embedder_batch_size: Размер батча для embedder (должен быть = 1)
            embedder_max_tokens: Максимальное количество токенов для embedder
        """
        self.llm_url = f"{llm_endpoint}/chat/completions"
        self.embedder_url = f"{embedder_endpoint}/embeddings"
        self.llm_model = llm_model
        self.embedder_model = embedder_model
        self.llm_max_retries = llm_max_retries
        self.llm_timeout = llm_timeout
        self.embedder_max_retries = embedder_max_retries
        self.embedder_timeout = embedder_timeout
        self.embedder_batch_size = embedder_batch_size
        self.embedder_max_tokens = embedder_max_tokens
        
        # Инициализируем токенайзер для embedder модели (один раз для всех экземпляров)
        if APIClient._tokenizer is None:
            logger.info(f"Загрузка токенайзера для {embedder_model}...")
            APIClient._tokenizer = AutoTokenizer.from_pretrained(embedder_model)
            logger.info("Токенайзер загружен")

        # Предупреждение если batch_size > 1
        if self.embedder_batch_size > 1:
            logger.warning(
                f"Embedder batch_size={self.embedder_batch_size}, но сервер настроен на "
                f"--max-num-seqs={EMBEDDER_CONCURRENT_REQUESTS}. Это может привести к ошибкам!"
            )

    def _calculate_retry_delay(
        self,
        attempt: int,
        base_delay: float,
        max_delay: float,
        jitter: float,
    ) -> float:
        """
        Вычисляет задержку для retry с экспоненциальным backoff и jitter.

        Args:
            attempt: Номер попытки (начиная с 1)
            base_delay: Базовая задержка в секундах
            max_delay: Максимальная задержка в секундах
            jitter: Коэффициент случайного разброса (0-1)

        Returns:
            Задержка в секундах
        """
        # Экспоненциальная задержка: base_delay * 2^(attempt-1)
        exponential_delay = base_delay * (2 ** (attempt - 1))

        # Ограничиваем максимальной задержкой
        delay = min(exponential_delay, max_delay)

        # Добавляем случайный разброс (jitter)
        jitter_amount = delay * jitter * random.uniform(-1, 1)
        delay_with_jitter = delay + jitter_amount

        # Гарантируем положительную задержку
        return max(0.0, delay_with_jitter)

    def _split_text_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        """
        Разделяет текст на части по max_tokens токенов.
        
        ВАЖНО: Учитывает, что сервер добавит special tokens (CLS, SEP и т.д.),
        поэтому используем add_special_tokens=True при подсчете.
        
        Args:
            text: Текст для разделения
            max_tokens: Максимальное количество токенов в части
            
        Returns:
            Список частей текста
        """
        # Токенизируем весь текст С учетом special tokens
        tokens = APIClient._tokenizer.encode(text, add_special_tokens=True)
        
        # Если текст короткий, возвращаем как есть
        if len(tokens) <= max_tokens:
            return [text]
        
        # Разбиваем на части
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = APIClient._tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
        
        logger.debug(
            f"Текст разделен на {len(chunks)} частей "
            f"(исходная длина: {len(tokens)} токенов, max: {max_tokens})"
        )
        
        return chunks

    def _retry_request(
        self,
        func: Callable[..., Any],
        max_retries: int,
        base_delay: float,
        max_delay: float,
        jitter: float,
        operation_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Выполняет запрос с повторами при ошибке.

        Args:
            func: Функция для выполнения
            max_retries: Максимальное количество повторов
            base_delay: Базовая задержка между попытками
            max_delay: Максимальная задержка между попытками
            jitter: Коэффициент случайного разброса
            operation_name: Название операции для логирования
            *args: Позиционные аргументы для func
            **kwargs: Именованные аргументы для func

        Returns:
            Результат выполнения функции

        Raises:
            Exception: Если все попытки исчерпаны
        """
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"{operation_name}: Попытка {attempt}/{max_retries} не удалась: {e}"
                )

                if attempt < max_retries:
                    delay = self._calculate_retry_delay(
                        attempt=attempt,
                        base_delay=base_delay,
                        max_delay=max_delay,
                        jitter=jitter,
                    )
                    logger.info(
                        f"{operation_name}: Ожидание {delay:.2f}с перед следующей попыткой"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"{operation_name}: Все {max_retries} попыток исчерпаны"
                    )

        if last_error:
            raise last_error

        raise RuntimeError(f"{operation_name}: Не удалось выполнить запрос")

    def generate(
        self,
        prompt: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        system_prompt: str | None = None,
    ) -> str:
        """
        Генерирует текст с помощью LLM с автоматическими retry.

        Args:
            prompt: Текстовый промпт
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            system_prompt: Системный промпт (опционально)

        Returns:
            Сгенерированный текст

        Raises:
            Exception: Если все попытки исчерпаны
        """

        def _generate() -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.llm_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            response = requests.post(
                self.llm_url, json=payload, timeout=self.llm_timeout
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        return self._retry_request(
            func=_generate,
            max_retries=self.llm_max_retries,
            base_delay=LLM_RETRY_BASE_DELAY_SECONDS,
            max_delay=LLM_RETRY_MAX_DELAY_SECONDS,
            jitter=LLM_RETRY_JITTER,
            operation_name="LLM generate",
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Создает эмбеддинги для текстов с автоматическими retry.

        ВАЖНО: Сервер embedder настроен на --max-num-seqs=1, поэтому
        рекомендуется отправлять по 1 тексту за раз через embed_single().

        Если отправляется batch > 1, сервер обработает их последовательно,
        что может быть медленнее чем отправка отдельных запросов.

        Args:
            texts: Список текстов (рекомендуется len(texts) <= embedder_batch_size)

        Returns:
            Список векторов эмбеддингов

        Raises:
            Exception: Если все попытки исчерпаны
        """
        # Предупреждение о неоптимальном использовании
        if len(texts) > self.embedder_batch_size:
            logger.warning(
                f"Отправка {len(texts)} текстов в одном запросе, но batch_size={self.embedder_batch_size}. "
                f"Сервер обработает их последовательно. Рекомендуется использовать embed_single()."
            )
        
        # Обрабатываем каждый текст: если длинный - режем и усредняем
        result_embeddings = []
        
        for idx, text in enumerate(texts):
            # Разбиваем текст на части, если он превышает лимит
            text_chunks = self._split_text_by_tokens(text, self.embedder_max_tokens)
            
            if len(text_chunks) > 1:
                logger.info(
                    f"Текст {idx+1}/{len(texts)} разделен на {len(text_chunks)} частей "
                    f"(превышает {self.embedder_max_tokens} токенов)"
                )
            
            # Получаем эмбеддинги для каждой части
            chunk_embeddings = []
            
            def _embed_chunk(chunk: str) -> list[float]:
                payload = {
                    "model": self.embedder_model,
                    "input": [chunk],  # Всегда отправляем одну часть
                }

                response = requests.post(
                    self.embedder_url, json=payload, timeout=self.embedder_timeout
                )
                
                # Улучшенная обработка ошибок с выводом тела ответа
                try:
                    response.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    try:
                        error_body = response.json()
                        logger.error(f"Embedder API error: {error_body}")
                    except Exception:
                        logger.error(f"Embedder API error (raw): {response.text}")
                    raise e
                
                data = response.json()["data"]
                return data[0]["embedding"]
            
            for chunk in text_chunks:
                chunk_emb = self._retry_request(
                    func=_embed_chunk,
                    max_retries=self.embedder_max_retries,
                    base_delay=EMBEDDER_RETRY_BASE_DELAY_SECONDS,
                    max_delay=EMBEDDER_RETRY_MAX_DELAY_SECONDS,
                    jitter=EMBEDDER_RETRY_JITTER,
                    operation_name="Embedder embed",
                    chunk=chunk,
                )
                chunk_embeddings.append(chunk_emb)
            
            # Усредняем эмбеддинги всех частей
            if len(chunk_embeddings) == 1:
                final_embedding = chunk_embeddings[0]
            else:
                # Усредняем: суммируем и делим на количество
                final_embedding = np.mean(chunk_embeddings, axis=0).tolist()
                logger.debug(f"Усреднено {len(chunk_embeddings)} эмбеддингов для текста {idx+1}")
            
            result_embeddings.append(final_embedding)
        
        return result_embeddings

    def embed_single(self, text: str) -> list[float]:
        """
        Создает эмбеддинг для одного текста.

        Args:
            text: Текст

        Returns:
            Вектор эмбеддинга

        Raises:
            Exception: Если все попытки исчерпаны
        """
        return self.embed([text])[0]

    # ========================================================================
    # АСИНХРОННЫЕ МЕТОДЫ
    # ========================================================================

    async def generate_async(
        self,
        prompt: str,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        system_prompt: str | None = None,
    ) -> str:
        """
        Асинхронная генерация текста с автоматическим ограничением параллелизма.
        
        ВАЖНО: Использует семафор для ограничения до LLM_CONCURRENT_REQUESTS (8)
        параллельных запросов, что соответствует --max-num-seqs 8 на сервере.

        Args:
            prompt: Текстовый промпт
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            system_prompt: Системный промпт

        Returns:
            Сгенерированный текст

        Raises:
            Exception: Если все попытки исчерпаны
        """
        semaphore = _get_llm_semaphore()
        
        async with semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.generate,
                prompt,
                temperature,
                max_tokens,
                system_prompt,
            )

    async def embed_single_async(self, text: str) -> list[float]:
        """
        Асинхронное создание эмбеддинга для одного текста.
        
        ВАЖНО: Использует семафор для ограничения до EMBEDDER_CONCURRENT_REQUESTS (1)
        параллельных запросов, что соответствует --max-num-seqs 1 на сервере.
        
        Параллелизм = 1, поэтому запросы будут выполняться последовательно,
        но это позволяет единообразно работать с API в асинхронном коде.

        Args:
            text: Текст

        Returns:
            Вектор эмбеддинга

        Raises:
            Exception: Если все попытки исчерпаны
        """
        semaphore = _get_embedder_semaphore()
        
        async with semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.embed_single,
                text,
            )

    async def generate_batch_async(
        self,
        prompts: list[str],
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        system_prompt: str | None = None,
        show_progress: bool = True,
    ) -> list[str]:
        """
        Параллельная генерация для батча промптов.
        
        Автоматически ограничивает количество параллельных запросов
        до LLM_CONCURRENT_REQUESTS (8), что соответствует --max-num-seqs на сервере.

        Args:
            prompts: Список промптов
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            system_prompt: Системный промпт
            show_progress: Показывать прогресс в логах

        Returns:
            Список сгенерированных текстов

        Example:
            >>> async def process():
            >>>     client = APIClient()
            >>>     prompts = ["Вопрос 1", "Вопрос 2", ...]
            >>>     results = await client.generate_batch_async(prompts)
            >>>     return results
            >>>
            >>> results = asyncio.run(process())
        """
        total = len(prompts)
        logger.info(
            f"Начало параллельной генерации для {total} промптов "
            f"(max {LLM_CONCURRENT_REQUESTS} одновременно)"
        )

        completed = 0

        async def _generate_with_progress(idx: int, prompt: str) -> str:
            nonlocal completed
            result = await self.generate_async(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
            completed += 1
            if show_progress and completed % 10 == 0:
                logger.info(f"Обработано {completed}/{total} промптов")
            return result

        tasks = [_generate_with_progress(i, p) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Обработка ошибок
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        error_count = len(results) - success_count
        
        if error_count > 0:
            logger.warning(
                f"Завершено: {success_count} успешно, {error_count} ошибок"
            )
            # Для ошибок возвращаем пустую строку
            results = [r if not isinstance(r, Exception) else "" for r in results]
        else:
            logger.info(f"Все {success_count} промптов обработаны успешно")

        return results  # type: ignore

    async def embed_batch_async(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Параллельная генерация эмбеддингов для батча текстов.
        
        ВАЖНО: Несмотря на "параллельность", семафор ограничивает до
        EMBEDDER_CONCURRENT_REQUESTS (1), поэтому запросы будут выполняться
        последовательно. Это правильное поведение для --max-num-seqs 1.

        Args:
            texts: Список текстов
            show_progress: Показывать прогресс в логах

        Returns:
            Список векторов эмбеддингов

        Example:
            >>> async def process():
            >>>     client = APIClient()
            >>>     texts = ["Текст 1", "Текст 2", ...]
            >>>     embeddings = await client.embed_batch_async(texts)
            >>>     return embeddings
            >>>
            >>> embeddings = asyncio.run(process())
        """
        total = len(texts)
        logger.info(
            f"Начало создания эмбеддингов для {total} текстов "
            f"(max {EMBEDDER_CONCURRENT_REQUESTS} одновременно)"
        )

        completed = 0

        async def _embed_with_progress(idx: int, text: str) -> list[float]:
            nonlocal completed
            result = await self.embed_single_async(text)
            completed += 1
            if show_progress and completed % 100 == 0:
                logger.info(f"Обработано {completed}/{total} текстов")
            return result

        tasks = [_embed_with_progress(i, t) for i, t in enumerate(texts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Обработка ошибок
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        error_count = len(results) - success_count
        
        if error_count > 0:
            logger.warning(
                f"Завершено: {success_count} успешно, {error_count} ошибок"
            )
            # Для ошибок возвращаем пустой вектор
            results = [r if not isinstance(r, Exception) else [] for r in results]
        else:
            logger.info(f"Все {success_count} эмбеддингов созданы успешно")

        return results  # type: ignore


class RerankerClient:
    """Клиент для работы с Reranker через OpenAI-совместимый API."""

    def __init__(self, base_url: str | None = None) -> None:
        """
        Инициализация клиента Reranker.

        Args:
            base_url: Базовый URL для API Reranker
        """
        self.base_url = base_url or RERANKER_ENDPOINT
        self.rerank_url = f"{self.base_url}/rerank"

    def rerank(
        self,
        query: str,
        documents: list[str],
        model: str = RERANKER_MODEL,
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Переранжирует документы по релевантности к запросу.

        Args:
            query: Поисковый запрос
            documents: Список документов для ранжирования
            model: Название модели (по умолчанию из settings)
            top_n: Количество топ документов для возврата
                (если None, возвращаются все)

        Returns:
            Список словарей с полями:
            - index: индекс документа в исходном списке
            - score: оценка релевантности
            - text: текст документа

        Raises:
            requests.RequestException: При ошибке запроса
        """
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        logger.info(
            f"Reranker: query='{query[:50]}...', {len(documents)} документов"
        )
        
        try:
            response = requests.post(
                self.rerank_url,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()

            # Извлекаем результаты и нормализуем формат
            results = result.get("results", [])
            normalized_results = []
            for item in results:
                # Reranker возвращает relevance_score, а не score
                normalized_results.append(
                    {
                        "index": item.get("index", -1),
                        "score": item.get("relevance_score", 0.0),
                        "text": item.get("document", {}).get("text", ""),
                    }
                )

            logger.info(f"Reranker: {len(normalized_results)} результатов")
            return normalized_results
            
        except requests.RequestException as e:
            logger.error(f"Ошибка Reranker: {e}")
            raise
