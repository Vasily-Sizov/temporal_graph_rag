"""Настройки для Temporal Graph RAG."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
env_path = Path(".env")
if not env_path.exists():
    # Пробуем найти в корне проекта (2 уровня вверх от settings.py)
    env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
else:
    # Если .env не найден, пробуем загрузить из текущей директории (по умолчанию)
    load_dotenv(override=True)


# ============================================================================
# Настройки API серверов
# ============================================================================

# IP адрес сервера
SERVER_IP: str = os.getenv("IP") or "localhost"

# Порты для сервисов
LLM_PORT: int = 8001
EMBEDDER_PORT: int = 8006
RERANKER_PORT: int = 8010

# Endpoints для API
LLM_ENDPOINT: str = f"http://{SERVER_IP}:{LLM_PORT}/v1"
EMBEDDER_ENDPOINT: str = f"http://{SERVER_IP}:{EMBEDDER_PORT}/v1"
RERANKER_ENDPOINT: str = f"http://{SERVER_IP}:{RERANKER_PORT}/v1"

# ============================================================================
# Настройки моделей
# ============================================================================

# Названия моделей по умолчанию
LLM_MODEL: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
EMBEDDER_MODEL: str = "deepvk/USER-bge-m3"
RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

# Температура генерации для LLM
LLM_TEMPERATURE: float = 0.0
LLM_MAX_TOKENS: int = 4096
"""Максимальное количество токенов в ответе LLM."""

LLM_CONTEXT_WINDOW: int = 16384
"""Размер контекстного окна модели."""

LLM_CONTEXT_SAFETY_MARGIN: int = 256
"""Запас токенов для безопасности."""

# ============================================================================
# Настройки retry механизма
# ============================================================================

# Таймауты для запросов (в секундах)
REQUEST_TIMEOUT: int = 300  # 5 минут

LLM_REQUEST_TIMEOUT: int = 300
"""Таймаут для запросов к LLM."""

LLM_MAX_RETRIES: int = 5
"""Максимальное количество повторных попыток для LLM запросов."""

LLM_RETRY_BASE_DELAY_SECONDS: float = 5.0
"""Базовая задержка между повторными попытками."""

LLM_RETRY_MAX_DELAY_SECONDS: float = 60.0
"""Максимальная задержка между повторными попытками."""

LLM_RETRY_JITTER: float = 0.2
"""Случайный разброс для задержки (jitter)."""

EMBEDDER_REQUEST_TIMEOUT: int = 300
"""Таймаут для запросов к Embedder."""

EMBEDDER_MAX_RETRIES: int = 5
"""Максимальное количество повторных попыток для Embedder запросов."""

EMBEDDER_RETRY_BASE_DELAY_SECONDS: float = 5.0
"""Базовая задержка между повторными попытками для Embedder."""

EMBEDDER_RETRY_MAX_DELAY_SECONDS: float = 60.0
"""Максимальная задержка между повторными попытками для Embedder."""

EMBEDDER_RETRY_JITTER: float = 0.2
"""Случайный разброс для задержки Embedder (jitter)."""

# ============================================================================
# Настройки параллелизма (соответствуют --max-num-seqs в docker-compose)
# ============================================================================

LLM_CONCURRENT_REQUESTS: int = 8
"""Количество параллельных запросов к LLM. Должно соответствовать --max-num-seqs на сервере."""

EMBEDDER_CONCURRENT_REQUESTS: int = 1
"""Количество параллельных запросов к Embedder. Должно соответствовать --max-num-seqs на сервере."""

# ============================================================================
# Настройки эмбеддингов
# ============================================================================

EMBEDDER_MAX_TOKENS: int = 509
"""Максимальная длина входа для модели эмбеддингов (512 - 3 для special tokens)."""

EMBEDDER_BATCH_SIZE: int = 1
"""Размер батча для эмбеддингов. ВАЖНО: Должно быть = 1, так как --max-num-seqs 1 на сервере."""

EMBEDDER_BATCH_MAX_TOKENS: int = EMBEDDER_MAX_TOKENS - 112
"""Максимальное количество токенов в батче для эмбеддингов (512-112)."""

# Создание эмбеддингов для разных типов данных
CREATE_ENTITY_EMBEDDINGS: bool = True
"""Создавать ли эмбеддинги для entities (нужно для entity-centric search)."""

CREATE_COMMUNITY_EMBEDDINGS: bool = True
"""Создавать ли эмбеддинги для communities (нужно для community-centric search)."""

CREATE_TEXT_UNIT_EMBEDDINGS: bool = True
"""
Создавать ли эмбеддинги для text_units (сырых чанков).
Полезно для сравнения с классическим RAG, но увеличивает размер индекса и время построения.
"""

# ============================================================================
# Настройки Temporal Graph RAG
# ============================================================================

# Извлечение графа
EXTRACT_GRAPH_TEMPERATURE: float = 0.0
"""Температура для извлечения графа."""

EXTRACT_GRAPH_MAX_TOKENS: int = 4096
"""Максимальное количество токенов для извлечения графа."""

# Генерация отчетов сообществ
COMMUNITY_REPORT_TEMPERATURE: float = 0.0
"""Температура для генерации отчетов сообществ."""

COMMUNITY_REPORT_MAX_TOKENS: int = LLM_MAX_TOKENS - LLM_CONTEXT_SAFETY_MARGIN
"""Максимальное количество токенов в отчете сообщества."""

COMMUNITY_REPORT_MAX_INPUT_LENGTH: int = (
    LLM_CONTEXT_WINDOW - LLM_MAX_TOKENS - LLM_CONTEXT_SAFETY_MARGIN
)
"""Максимальная длина входного контекста для генерации отчетов сообществ."""

# Контекст для поиска
SEARCH_MAX_CONTEXT_TOKENS: int = (
    LLM_CONTEXT_WINDOW - LLM_MAX_TOKENS - LLM_CONTEXT_SAFETY_MARGIN
)
"""Максимальное количество токенов в контексте для поиска."""

# ============================================================================
# Настройки кластеризации
# ============================================================================

CLUSTERING_RESOLUTION: float = 1.0
"""Разрешение для алгоритма кластеризации Leiden."""

MAX_COMMUNITY_LEVELS: int = 3
"""Максимальное количество уровней иерархии сообществ."""

MAX_CLUSTER_SIZE: int = 50
"""Максимальный размер кластера."""

USE_LCC: bool = True
"""Использовать Largest Connected Component для кластеризации."""

# ============================================================================
# Настройки поиска
# ============================================================================

DEFAULT_TOP_K_COMMUNITIES: int = 5
"""Количество сообществ для поиска по умолчанию."""

LOCAL_SEARCH_TEXT_UNIT_PROP: float = 0.7
"""Доля текстовых единиц в контексте Local Search."""

LOCAL_SEARCH_COMMUNITY_PROP: float = 0.3
"""Доля отчетов сообществ в контексте Local Search."""

# ============================================================================
# Настройки классического RAG
# ============================================================================

CLASSIC_RAG_TOP_K_SEARCH: int = 50
"""Количество документов для векторного поиска в классическом RAG."""

CLASSIC_RAG_TOP_K_RERANK: int = 5
"""Количество документов после reranking в классическом RAG."""

CLASSIC_RAG_USE_RERANKER: bool = True
"""Использовать ли reranker в классическом RAG."""

# ============================================================================
# Настройки временного окна для communities
# ============================================================================

DEFAULT_COMMUNITY_WINDOW_SIZE: int = 20
"""Размер временного окна для communities в чанках по умолчанию."""

DEFAULT_COMMUNITY_OVERLAP: int = 5
"""Перекрытие между community окнами в чанках по умолчанию."""

# ============================================================================
# Настройки checkpoints для построения индекса
# ============================================================================

CHECKPOINT_FREQUENCY: int = 20
"""Частота сохранения checkpoints при извлечении графа (каждые N чанков)."""

ENABLE_CHECKPOINTS: bool = True
"""Включить систему checkpoints для восстановления после сбоев."""

# ============================================================================
# Настройки парсинга и чанкинга документов
# ============================================================================

# Парсинг PDF
PDF_DO_OCR: bool = False
"""Использовать ли OCR при парсинге PDF (False = только текстовый слой)."""

PDF_DO_TABLE_STRUCTURE: bool = False
"""Обрабатывать ли структуру таблиц при парсинге PDF."""

PDF_FILTER_TABLE_OF_CONTENTS: bool = True
"""Фильтровать ли оглавление (list_item элементы в начале документа)."""

PDF_TOC_MAX_PAGE: int = 5
"""Максимальная страница для поиска оглавления."""

# Чанкинг
CHUNKING_EMBED_MODEL_ID: str = EMBEDDER_MODEL
"""ID модели для токенизатора при чанкинге (должна совпадать с моделью эмбеддингов)."""

CHUNKING_MAX_TOKENS: int = 512
"""Максимальное количество токенов в чанке."""

CHUNKING_METHOD: str = "sentence"
"""Метод чанкинга: 'hybrid' (Docling) или 'sentence' (на основе предложений)."""

SENTENCE_CHUNKING_MAX_TOKENS: int = 510
"""Максимальное количество токенов в чанке для sentence-based chunker (рекомендуется 256-510 для BAAI/bge). Чанки не пересекают границы глав."""
