# Temporal Graph RAG

Temporal Graph RAG — фреймворк для построения временных графов знаний по серии книг (на примере «Гарри Поттера»). Сравнивает два подхода: **Classic RAG** (FAISS + Reranker) и **Temporal Graph RAG** (граф сущностей со скользящими окнами сообществ).

**Модели** (все через vLLM):
- LLM: `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`
- Embedder: `deepvk/USER-bge-m3`
- Reranker: `BAAI/bge-reranker-v2-m3`

---

## Требования

- Docker + Docker Compose
- GPU-сервер с NVIDIA GPU (для vLLM)
- Файл `.env` с переменной `IP` (адрес GPU-сервера)

```bash
# .env
IP=192.168.1.100
```

---

## 1. Запуск моделей (GPU-сервер)

На сервере с GPU запустить три модели через vLLM:

```bash
cd vllm_llm_embedder_reranker
docker compose up -d
```

Это поднимет:

| Сервис | Порт | Модель | GPU Memory |
|--------|------|--------|------------|
| LLM | 8001 | Qwen3-30B-A3B-FP8 | 80% |
| Embedder | 8006 | deepvk/USER-bge-m3 | 10% |
| Reranker | 8010 | BAAI/bge-reranker-v2-m3 | 15% |

Проверить доступность:

```bash
curl http://$IP:8001/v1/models
curl http://$IP:8006/v1/models
curl http://$IP:8010/v1/models
```

---

## 2. Сборка проекта

```bash
docker compose build temporal-graph-rag
```

---

## 3. Парсинг PDF и подготовка чанков

PDF-файлы книг должны лежать в `../input_files/`. Парсинг и чанкинг выполняются из контейнера:

```bash
# Парсинг PDF → JSON (через Docling)
docker compose run --rm temporal-graph-rag \
  python -m temporal_graph_rag.document_processing.pdf_parser \
  /app/data/input_files \
  /app/data/output/parsing_and_chunks/parsed

# Чанкинг JSON → chunks JSON (sentence-based, макс. 510 токенов)
docker compose run --rm temporal-graph-rag \
  python -m temporal_graph_rag.document_processing.sentence_based_chunker \
  /app/data/output/parsing_and_chunks/parsed \
  /app/data/output/parsing_and_chunks/chunks
```

---

## 4. Temporal Graph RAG

### 4.1. Построение индекса

Через docker compose profile:

```bash
docker compose --profile build run --rm build-index
```

Или напрямую через CLI с параметрами:

```bash
docker compose run --rm temporal-graph-rag \
  python -m temporal_graph_rag.cli build \
  /app/data/output/parsing_and_chunks/chunks \
  /app/data/output/temporal_index \
  --community-window-size 20 \
  --community-overlap 5
```

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `chunks_dir` | — | Директория с JSON-файлами чанков |
| `output_dir` | — | Куда сохранить индекс |
| `--community-window-size` | 20 | Размер скользящего окна (в чанках) |
| `--community-overlap` | 5 | Перекрытие между окнами (в чанках) |

Индексация включает 5 шагов с чекпоинтами (каждые 20 чанков): извлечение графа → эмбеддинги сущностей → построение FAISS → сообщества → эмбеддинги сообществ.

### 4.2. Поиск (Temporal Local Search)

Через docker compose profile:

```bash
docker compose --profile search run --rm search
```

Или напрямую:

```bash
docker compose run --rm temporal-graph-rag \
  python -m temporal_graph_rag.cli search \
  /app/data/output/temporal_index \
  "Как изменился Гарри Поттер за серию?" \
  --books 1,7 \
  --top-k 5 \
  --top-k-entities 10 \
  --output /app/data/output/queries/result.md
```

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `index_dir` | — | Директория с индексом |
| `query` | — | Поисковый запрос |
| `--books` | все | Фильтр по книгам: `1,2,3` |
| `--top-k` | 5 | Кол-во communities для контекста |
| `--top-k-entities` | 10 | Кол-во entities для поиска |
| `--output` / `-o` | — | Путь для сохранения результата (Markdown) |

---

## 5. Classic RAG

### 5.1. Поиск (Classic RAG)

Индекс FAISS для text_units строится автоматически при построении основного индекса (шаг 3). Для отдельного поиска:

```bash
docker compose run --rm temporal-graph-rag \
  python -m temporal_graph_rag.cli classic_search \
  /app/data/output/temporal_index \
  "Как Квирелл узнал, как пройти мимо Пушка??" \
  --top-k-search 50 \
  --top-k-rerank 5 \
  --use-reranker \
  --output /app/data/output/classic_queries/classic_result.md
```

**Параметры:**

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `index_dir` | — | Директория с индексом (нужны `text_units.faiss` и `text_units.json`) |
| `query` | — | Поисковый запрос |
| `--top-k-search` | 50 | Сколько чанков достать из FAISS |
| `--top-k-rerank` | 5 | Сколько оставить после reranker |
| `--use-reranker` | да | Включить reranker |
| `--no-reranker` | — | Отключить reranker |
| `--output` / `-o` | — | Путь для сохранения результата |

---

## Примеры запросов

```bash
# Temporal Graph RAG — цепочка событий (нужен граф связей)
docker compose run --rm temporal-graph-rag \
  python -m temporal_graph_rag.cli search \
  /app/data/output/temporal_index \
  "Какая цепочка событий привела к тому, что Квиррелл узнал способ успокоить Пушка?"

docker compose run --rm temporal-graph-rag python -m temporal_graph_rag.cli search /app/data/output/temporal_index "Как Квирелл узнал, как пройти мимо Пушка?" --output /app/data/output/temporal_queries/result.md

# Temporal Graph RAG — перечисление с фильтром по книге
docker compose run --rm temporal-graph-rag \
  python -m temporal_graph_rag.cli search \
  /app/data/output/temporal_index \
  "Перечисли всех учителей Хогвартса и их предметы" \
  --books 1

# Classic RAG — точечный факт
docker compose run --rm temporal-graph-rag \
  python -m temporal_graph_rag.cli classic_search \
  /app/data/output/temporal_index \
  "Какой номер у платформы, с которой отправляется поезд в Хогвартс?"
```

---

## Тестовые вопросы

1. Как пройти мимо Пушка?
2. Перечисли всех учителей Хогвартса и их предметы, которые они преподают
3. Перечисли все факультеты и их описание
4. Какая цепочка событий, начавшаяся с действий Хагрида, в итоге привела к тому, что Квиррелл узнал способ успокоить Пушка?
5. Сопоставьте ученика и факультет, на который его определила Распределяющая шляпа: А. Ханна Аббот Б. Блейз Забини В. Терри Бут Г. Джастин Финч-Флетчли
6. Какие препятствия последовательно проходят Гарри, Рон и Гермиона на пути к Философскому камню и кто из них какое препятствие преодолевает?
7. Рон, Гермиона, Гарри преодолевали препятствия после прохождения Пушка?
8. Кто знал, как пройти мимо Пушка?
9. Какой номер у платформы, с которой отправляется поезд в Хогвартс?
10. Сколько денег у Гарри в банке Гринготтс?
11. Какую фирму возглавлял Дурсль?
12. Как зовут отца Гарри Поттера?
13. Кто упоминал Философский Камень, но не имел к нему доступа?
14. Как изменился Гарри Поттер за серию?

---

## Структура проекта

```
temporal_graph_rag/
├── cli.py                        # CLI: build, search, classic_search
├── settings.py                   # Все настройки (модели, порты, retry, лимиты)
├── models/
│   └── data_models.py            # Dataclass-ы: Entity, Relationship, Community, TextUnit
├── document_processing/
│   ├── cli.py                    # CLI для парсинга и чанкинга
│   ├── pdf_parser.py             # Парсинг PDF через Docling
│   ├── chunker.py                # Базовый чанкер
│   └── sentence_based_chunker.py # Чанкинг по предложениям
├── indexing/
│   ├── index_builder.py          # Оркестратор 5 шагов с чекпоинтами
│   ├── graph_extractor.py        # Извлечение сущностей и связей через LLM
│   └── community_builder.py      # Скользящие окна → отчёты сообществ
├── search/
│   ├── temporal_local_search.py  # 6-шаговый entity-centric поиск
│   ├── classic_rag.py            # FAISS → Reranker → LLM
│   └── index_loader.py           # Загрузка индекса из Parquet + FAISS
├── storage/
│   ├── parquet_storage.py        # Parquet + Snappy для табличных данных
│   └── vector_storage.py         # FAISS IndexFlatL2 для эмбеддингов
├── prompts/
│   ├── extract_graph_prompt.py    # Промпт извлечения графа
│   ├── community_report_prompt.py # Промпт отчётов сообществ
│   └── classic_rag_prompt.py      # Промпт классического RAG
└── utils/
    ├── api_client.py              # HTTP-клиент с retry и семафорами
    ├── chunk_loader.py            # Загрузка и нормализация чанков
    ├── id_utils.py                # MD5 детерминированные ID
    └── result_formatter.py        # Форматирование результатов в Markdown

vllm_llm_embedder_reranker/       # Docker Compose для моделей на GPU-сервере
├── docker-compose.yml
└── README.md
```
