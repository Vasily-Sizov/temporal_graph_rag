"""Чанкер документов с использованием HybridChunker из docling."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import DoclingDocument
from transformers import AutoTokenizer

from temporal_graph_rag import settings

logger = logging.getLogger(__name__)


@dataclass
class ChunkData:
    """Структура данных для чанка."""

    chunk_id: str
    text: str
    contextualized_text: str
    text_tokens: int
    contextualized_tokens: int
    metadata: dict[str, Any]


class DocumentChunker:
    """
    Чанкер документов с использованием HybridChunker.
    
    HybridChunker:
    - Сохраняет структуру документа (абзацы, заголовки)
    - Режет большие элементы до max_tokens
    - Объединяет маленькие элементы с одинаковыми headings
    - Правильно обрабатывает переходы между главами
    """

    def __init__(
        self,
        embed_model_id: str = settings.CHUNKING_EMBED_MODEL_ID,
        max_tokens: int = settings.CHUNKING_MAX_TOKENS,
        merge_peers: bool = True,
    ) -> None:
        """
        Инициализация чанкера.

        Args:
            embed_model_id: ID модели для embedding и токенизации
            max_tokens: Максимальное количество токенов в чанке
            merge_peers: Объединять ли соседние чанки с одинаковыми headings
        """
        logger.info(f"Инициализация DocumentChunker (HybridChunker):")
        logger.info(f"  Модель: {embed_model_id}")
        logger.info(f"  Макс. токенов: {max_tokens}")
        logger.info(f"  Объединение соседних: {merge_peers}")

        try:
            # Загрузка tokenizer для модели
            hf_tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
            tokenizer = HuggingFaceTokenizer(
                tokenizer=hf_tokenizer,
                max_tokens=max_tokens,
            )

            # Создание HybridChunker - он сочетает структуру + токенизацию
            self.chunker = HybridChunker(
                tokenizer=tokenizer,
                merge_peers=merge_peers,
            )
            self.tokenizer = tokenizer
            self.max_tokens = max_tokens

            logger.info("DocumentChunker успешно инициализирован")

        except Exception as e:
            logger.error(f"Ошибка при инициализации чанкера: {e}", exc_info=True)
            raise

    def chunk_document(self, doc: DoclingDocument) -> Iterator[ChunkData]:
        """
        Разбивает документ на чанки.

        Args:
            doc: DoclingDocument для чанкинга

        Yields:
            ChunkData объекты с информацией о чанках
        """
        logger.info("Начинаю чанкинг документа...")

        chunk_iter = self.chunker.chunk(dl_doc=doc)
        chunk_count = 0

        for chunk in chunk_iter:
            chunk_count += 1

            # Получаем текст чанка
            chunk_text = chunk.text

            # Получаем обогащенный текст (с метаданными)
            contextualized_text = self.chunker.contextualize(chunk=chunk)

            # Подсчитываем токены
            text_tokens = self.tokenizer.count_tokens(chunk_text)
            contextualized_tokens = self.tokenizer.count_tokens(contextualized_text)

            # Извлекаем метаданные
            metadata: dict[str, Any] = {}
            if hasattr(chunk, "meta"):
                try:
                    if hasattr(chunk.meta, "model_dump"):
                        metadata = chunk.meta.model_dump()
                    elif hasattr(chunk.meta, "dict"):
                        metadata = chunk.meta.dict()
                    else:
                        metadata = {"raw": str(chunk.meta)}
                except Exception as e:
                    logger.warning(
                        f"Не удалось извлечь метаданные для чанка {chunk_count}: {e}"
                    )
                    metadata = {"error": str(e)}

            yield ChunkData(
                chunk_id=f"chunk_{chunk_count}",
                text=chunk_text,
                contextualized_text=contextualized_text,
                text_tokens=text_tokens,
                contextualized_tokens=contextualized_tokens,
                metadata=metadata,
            )

        logger.info(f"Чанкинг завершен. Создано чанков: {chunk_count}")

    def chunk_and_save(
        self,
        doc: DoclingDocument,
        output_path: str | Path,
    ) -> int:
        """
        Разбивает документ на чанки и сохраняет в JSON.

        Args:
            doc: DoclingDocument для чанкинга
            output_path: Путь для сохранения JSON файла

        Returns:
            Количество созданных чанков
        """
        chunks = self.chunk_document(doc)
        return self.save_chunks(chunks, output_path)

    def save_chunks(
        self,
        chunks: Iterator[ChunkData],
        output_path: str | Path,
    ) -> int:
        """
        Сохраняет чанки в JSON файл.

        Args:
            chunks: Итератор чанков
            output_path: Путь для сохранения JSON файла

        Returns:
            Количество сохраненных чанков
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        chunks_list = list(chunks)
        chunks_dict = [asdict(chunk) for chunk in chunks_list]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"Сохранено {len(chunks_list)} чанков в: {output_path}")
        return len(chunks_list)


def load_document_from_json(json_path: str | Path) -> DoclingDocument:
    """
    Загружает DoclingDocument из JSON файла.

    Args:
        json_path: Путь к JSON файлу с документом

    Returns:
        Загруженный DoclingDocument

    Raises:
        FileNotFoundError: Если файл не найден
        RuntimeError: Если произошла ошибка при загрузке
    """
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"Файл не найден: {json_path}")

    logger.info(f"Загрузка документа из: {json_path}")
    try:
        doc = DoclingDocument.load_from_json(str(json_file))
        logger.info("Документ успешно загружен")
        return doc
    except Exception as e:
        logger.error(f"Ошибка при загрузке документа: {e}", exc_info=True)
        raise RuntimeError(f"Не удалось загрузить документ: {e}") from e


def chunk_document_from_json(
    input_json: str | Path,
    output_json: str | Path,
    embed_model_id: str = settings.CHUNKING_EMBED_MODEL_ID,
    max_tokens: int = settings.CHUNKING_MAX_TOKENS,
    merge_peers: bool = True,
) -> int:
    """
    Удобная функция для чанкинга документа из JSON файла.

    Args:
        input_json: Путь к JSON файлу с обработанным документом
        output_json: Путь для сохранения чанков
        embed_model_id: ID модели для embedding
        max_tokens: Максимальное количество токенов в чанке
        merge_peers: Объединять ли соседние чанки с одинаковыми headings

    Returns:
        Количество созданных чанков
    """
    # Загрузка документа
    doc = load_document_from_json(input_json)

    # Создание чанкера
    chunker = DocumentChunker(
        embed_model_id=embed_model_id,
        max_tokens=max_tokens,
        merge_peers=merge_peers,
    )

    # Чанкинг и сохранение
    count = chunker.chunk_and_save(doc, output_json)

    # Статистика
    logger.info("=" * 60)
    logger.info("Статистика чанкинга:")
    logger.info(f"  Всего чанков: {count}")
    logger.info(f"  Модель: {embed_model_id}")
    logger.info(f"  Макс. токенов: {max_tokens}")
    logger.info("=" * 60)

    return count

