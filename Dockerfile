FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Установка uv
RUN pip install --no-cache-dir uv

# Установка переменных окружения для оптимизации uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Установка рабочей директории
WORKDIR /app

# Копирование файла зависимостей
COPY pyproject.toml ./

# Создание lock файла из pyproject.toml (генерирует uv.lock)
# Этот слой будет кешироваться, если pyproject.toml не изменился
RUN uv lock

# Копирование минимального кода для установки зависимостей (только __init__.py)
# Это нужно для uv pip install -e .
COPY temporal_graph_rag/__init__.py ./temporal_graph_rag/__init__.py

# Установка зависимостей через uv (используя созданный lock файл)
# Этот слой будет кешироваться, если pyproject.toml и uv.lock не изменились
RUN uv pip install --python /usr/local/bin/python --system -e .

# Копирование остального кода проекта (после установки зависимостей для лучшего кеширования)
# Этот слой будет пересобираться только при изменении кода
COPY temporal_graph_rag/ ./temporal_graph_rag/

# Переустановка проекта с полным кодом (быстро, так как зависимости уже установлены)
# Используем --no-deps чтобы не переустанавливать зависимости
RUN uv pip install --python /usr/local/bin/python --system -e . --no-deps

# Установка переменных окружения
ENV PYTHONUNBUFFERED=1

# Команда по умолчанию
CMD ["python", "--version"]

