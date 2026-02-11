"""Чанкер на основе предложений для художественных текстов."""

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from docling_core.types.doc.document import DoclingDocument
from transformers import AutoTokenizer

from temporal_graph_rag import settings

logger = logging.getLogger(__name__)


@dataclass
class SentenceChunk:
    """Структура данных для чанка на основе предложений."""

    chunk_id: str
    text: str
    tokens: int
    sentence_count: int
    page_numbers: list[int]
    headings: list[str]
    filename: str
    metadata: dict[str, Any]


class SentenceBasedChunker:
    """
    Чанкер на основе предложений для художественных текстов.
    
    Особенности:
    - Разбивает текст на предложения
    - Создает чанки заданного размера (в токенах)
    - Всегда режет по законченным предложениям
    - Чанки не пересекают границы заголовков (завершает чанк при смене любого заголовка)
    - Оптимизирован для художественных текстов со структурой глав
    """

    def __init__(
        self,
        embed_model_id: str = settings.CHUNKING_EMBED_MODEL_ID,
        max_tokens: int = 256,
    ) -> None:
        """
        Инициализация чанкера.

        Args:
            embed_model_id: ID модели для embedding и токенизации
            max_tokens: Максимальное количество токенов в чанке (рекомендуется 256-510)
        """
        logger.info("Инициализация SentenceBasedChunker:")
        logger.info(f"  Модель: {embed_model_id}")
        logger.info(f"  Макс. токенов: {max_tokens}")
        logger.info("  Чанки не пересекают границы глав")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
            self.max_tokens = max_tokens
            logger.info("✓ Токенизатор загружен успешно")
        except Exception as e:
            logger.error(f"Ошибка при загрузке токенизатора: {e}", exc_info=True)
            raise

    def split_into_sentences(self, text: str) -> list[str]:
        """
        Разбивает текст на предложения.

        Использует регулярные выражения для определения границ предложений.
        Учитывает особенности русского языка (аббревиатуры, инициалы и т.п.).

        Args:
            text: Текст для разбиения

        Returns:
            Список предложений
        """
        # Убираем множественные пробелы и переносы строк
        text = re.sub(r"\s+", " ", text).strip()

        # Паттерн для разбиения на предложения
        # Ищем точку, восклицательный или вопросительный знак, за которым следует пробел и заглавная буква
        # Или многоточие
        sentence_pattern = r"(?<=[.!?…])\s+(?=[А-ЯA-Z«\-\d])"

        sentences = re.split(sentence_pattern, text)

        # Убираем пустые предложения и лишние пробелы
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def count_tokens(self, text: str) -> int:
        """
        Подсчитывает количество токенов в тексте.

        Args:
            text: Текст для подсчета

        Returns:
            Количество токенов
        """
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def _create_chunks_with_metadata(
        self, sentences_with_meta: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Создает чанки из предложений с метаданными.
        
        Чанки не пересекают границы заголовков - при смене любого заголовка 
        текущий чанк автоматически завершается.

        Args:
            sentences_with_meta: Список словарей с предложениями и метаданными

        Returns:
            Список словарей с информацией о чанках
        """
        chunks: list[dict[str, Any]] = []
        current_sentences: list[str] = []
        current_pages: set[int] = set()
        current_headings: set[str] = set()
        current_filename: str = ""
        current_tokens = 0
        current_headings_tuple: tuple[str, ...] | None = None  # Текущие заголовки как кортеж

        def save_current_chunk() -> None:
            """Сохраняет текущий чанк и очищает буферы."""
            if current_sentences:
                chunks.append({
                    "text": " ".join(current_sentences),
                    "tokens": current_tokens,
                    "sentence_count": len(current_sentences),
                    "page_numbers": sorted(current_pages),
                    "headings": list(current_headings),
                    "filename": current_filename,
                })

        i = 0
        while i < len(sentences_with_meta):
            sent_data = sentences_with_meta[i]
            sentence = sent_data["text"]
            sentence_tokens = self.count_tokens(sentence)
            sent_headings_tuple = tuple(sent_data["headings"])

            # Проверяем смену заголовков
            headings_changed = (
                current_headings_tuple is not None 
                and sent_headings_tuple != current_headings_tuple
            )

            # Если изменились заголовки, завершаем текущий чанк
            if headings_changed:
                logger.debug(
                    f"Смена заголовков: {current_headings_tuple} -> {sent_headings_tuple}. "
                    "Завершаю чанк."
                )
                save_current_chunk()
                # Очищаем буферы
                current_sentences = []
                current_pages = set()
                current_headings = set()
                current_tokens = 0
                current_headings_tuple = sent_headings_tuple

            # Обновляем текущие заголовки, если они ещё не установлены
            if current_headings_tuple is None:
                current_headings_tuple = sent_headings_tuple

            # Если одно предложение больше max_tokens, режем его
            if sentence_tokens > self.max_tokens:
                logger.warning(
                    f"Предложение превышает max_tokens ({sentence_tokens} > {self.max_tokens}). "
                    "Разбиваю по словам..."
                )

                # Если уже есть накопленные предложения, сохраняем их
                save_current_chunk()
                current_sentences = []
                current_pages = set()
                current_headings = set()
                current_tokens = 0

                # Разбиваем длинное предложение по словам
                word_chunks = self._split_long_sentence(sentence, self.max_tokens)
                for word_chunk, tokens in word_chunks:
                    chunks.append({
                        "text": word_chunk,
                        "tokens": tokens,
                        "sentence_count": 1,
                        "page_numbers": sent_data["page_numbers"],
                        "headings": sent_data["headings"],
                        "filename": sent_data["filename"],
                    })

                i += 1
                continue

            # Проверяем, поместится ли предложение в текущий чанк
            test_text = " ".join(current_sentences + [sentence])
            test_tokens = self.count_tokens(test_text)

            if test_tokens <= self.max_tokens:
                # Добавляем предложение в текущий чанк
                current_sentences.append(sentence)
                current_tokens = test_tokens
                current_pages.update(sent_data["page_numbers"])
                current_headings.update(sent_data["headings"])
                current_filename = sent_data["filename"]
                i += 1
            else:
                # Текущий чанк заполнен, сохраняем его
                save_current_chunk()
                # Очищаем буферы для нового чанка
                current_sentences = []
                current_pages = set()
                current_headings = set()
                current_tokens = 0
                # Не увеличиваем i, чтобы обработать это предложение на следующей итерации

        # Сохраняем последний чанк
        save_current_chunk()

        return chunks

    def create_chunks_from_sentences(
        self, sentences: list[str]
    ) -> list[tuple[str, int, int]]:
        """
        Создает чанки из предложений с учетом лимита токенов.

        УСТАРЕЛО: Используйте _create_chunks_with_metadata для чанкинга с метаданными.
        Этот метод не учитывает границы глав.

        Args:
            sentences: Список предложений

        Returns:
            Список кортежей (текст_чанка, количество_токенов, количество_предложений)
        """
        chunks: list[tuple[str, int, int]] = []
        current_sentences: list[str] = []
        current_tokens = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.count_tokens(sentence)

            # Если одно предложение больше max_tokens, режем его
            if sentence_tokens > self.max_tokens:
                logger.warning(
                    f"Предложение превышает max_tokens ({sentence_tokens} > {self.max_tokens}). "
                    "Разбиваю по словам..."
                )

                # Если уже есть накопленные предложения, сохраняем их
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append((chunk_text, current_tokens, len(current_sentences)))
                    current_sentences = []
                    current_tokens = 0

                # Разбиваем длинное предложение по словам
                word_chunks = self._split_long_sentence(sentence, self.max_tokens)
                for word_chunk, tokens in word_chunks:
                    chunks.append((word_chunk, tokens, 1))

                i += 1
                continue

            # Проверяем, поместится ли предложение в текущий чанк
            test_text = " ".join(current_sentences + [sentence])
            test_tokens = self.count_tokens(test_text)

            if test_tokens <= self.max_tokens:
                # Добавляем предложение в текущий чанк
                current_sentences.append(sentence)
                current_tokens = test_tokens
                i += 1
            else:
                # Сохраняем текущий чанк
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    chunks.append((chunk_text, current_tokens, len(current_sentences)))
                    current_sentences = []
                    current_tokens = 0

        # Сохраняем последний чанк
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append((chunk_text, current_tokens, len(current_sentences)))

        return chunks

    def _split_long_sentence(
        self, sentence: str, max_tokens: int
    ) -> list[tuple[str, int]]:
        """
        Разбивает длинное предложение на части по словам.

        Args:
            sentence: Предложение для разбиения
            max_tokens: Максимальное количество токенов в части

        Returns:
            Список кортежей (текст_части, количество_токенов)
        """
        words = sentence.split()
        chunks: list[tuple[str, int]] = []
        current_words: list[str] = []
        current_tokens = 0

        for word in words:
            test_text = " ".join(current_words + [word])
            test_tokens = self.count_tokens(test_text)

            if test_tokens <= max_tokens:
                current_words.append(word)
                current_tokens = test_tokens
            else:
                if current_words:
                    chunk_text = " ".join(current_words)
                    chunks.append((chunk_text, current_tokens))
                    current_words = [word]
                    current_tokens = self.count_tokens(word)
                else:
                    # Одно слово больше max_tokens - сохраняем как есть
                    chunks.append((word, test_tokens))

        if current_words:
            chunk_text = " ".join(current_words)
            chunks.append((chunk_text, current_tokens))

        return chunks

    def chunk_text(self, text: str, filename: str = "") -> Iterator[SentenceChunk]:
        """
        Разбивает текст на чанки (без метаданных документа).

        УСТАРЕЛО: Используйте chunk_document для чанкинга с метаданными.
        Этот метод не учитывает границы глав.

        Args:
            text: Текст для чанкинга
            filename: Имя файла (опционально)

        Yields:
            SentenceChunk объекты с информацией о чанках
        """
        logger.info("Начинаю чанкинг текста...")

        # Разбиваем на предложения
        sentences = self.split_into_sentences(text)
        logger.info(f"  Разбито на {len(sentences)} предложений")

        # Создаем чанки
        chunks = self.create_chunks_from_sentences(sentences)
        logger.info(f"  Создано {len(chunks)} чанков")

        # Генерируем SentenceChunk объекты
        for i, (chunk_text, tokens, sentence_count) in enumerate(chunks, 1):
            yield SentenceChunk(
                chunk_id=f"chunk_{i}",
                text=chunk_text,
                tokens=tokens,
                sentence_count=sentence_count,
                page_numbers=[],
                headings=[],
                filename=filename,
                metadata={
                    "chunking_method": "sentence_based",
                    "max_tokens": self.max_tokens,
                },
            )

        logger.info("Чанкинг завершен")

    def _extract_elements_with_metadata(
        self, doc: DoclingDocument
    ) -> list[dict[str, Any]]:
        """
        Извлекает текстовые элементы из документа с метаданными.
        
        Автоматически склеивает незавершённые фрагменты (элементы, не заканчивающиеся
        на точку, восклицательный/вопросительный знак или многоточие).

        Args:
            doc: DoclingDocument для обработки

        Returns:
            Список словарей с информацией об элементах (text, page_no, heading, label)
        """
        raw_elements: list[dict[str, Any]] = []
        current_headings: list[str] = []

        # Получаем filename из origin
        filename = ""
        if hasattr(doc, "origin") and doc.origin:
            filename = getattr(doc.origin, "filename", "")

        # Итерируем по текстовым элементам через doc.texts
        if not hasattr(doc, "texts") or not doc.texts:
            logger.warning("Документ не содержит texts атрибут или он пустой")
            return []

        for item in doc.texts:
            # Получаем текст
            text = getattr(item, "text", "")
            if not text or not text.strip():
                continue

            # Получаем label (тип элемента)
            label = getattr(item, "label", "")

            # Обновляем текущие заголовки
            # Игнорируем page_header (разделители типа "* * *")
            if "header" in label.lower() and label != "page_header":
                level = getattr(item, "level", 1)
                text_stripped = text.strip()
                
                # ФИЛЬТРАЦИЯ: добавляем только настоящие главы
                # Пропускаем мусорные заголовки типа "Форма", подписи и т.д.
                # docling помечает как section_header любой жирный текст на отдельной строке,
                # включая подписи в письмах и подзаголовки в списках покупок
                is_valid_heading = (
                    text_stripped.startswith("Глава ") or  # Настоящие главы
                    text_stripped == "Annotation" or  # Аннотация
                    text_stripped.startswith("Часть ") or  # Части книги (если есть)
                    text_stripped.startswith("Эпилог") or  # Эпилог
                    text_stripped.startswith("Пролог")  # Пролог
                )
                
                if is_valid_heading:
                    # Обрезаем заголовки более высокого уровня
                    current_headings = current_headings[:level - 1]
                    current_headings.append(text_stripped)
                    logger.debug(f"Добавлен заголовок: '{text_stripped}' (level={level})")
                else:
                    logger.debug(f"Пропущен мусорный заголовок: '{text_stripped}' (label={label})")

            # Извлекаем номера страниц
            page_numbers: list[int] = []
            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    if hasattr(prov, "page_no"):
                        page_numbers.append(prov.page_no)

            # Убираем дубликаты и сортируем
            page_numbers = sorted(set(page_numbers))

            raw_elements.append({
                "text": text,
                "page_numbers": page_numbers,
                "headings": current_headings.copy(),
                "label": label,
                "filename": filename,
            })

        # Склеиваем незавершённые фрагменты
        merged_elements = self._merge_incomplete_elements(raw_elements)
        
        return merged_elements

    def _merge_incomplete_elements(
        self, elements: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Склеивает незавершённые текстовые фрагменты.
        
        Если элемент не заканчивается на завершающий знак препинания
        (точка, !, ?, многоточие), он склеивается со следующим элементом.
        
        Важно: склеивание происходит только для ОДНОГО незавершённого фрагмента,
        после чего начинается новый элемент, чтобы избежать накопления page_numbers.

        Args:
            elements: Список элементов с метаданными

        Returns:
            Список элементов со склеенными фрагментами
        """
        if not elements:
            return []

        merged: list[dict[str, Any]] = []
        i = 0

        while i < len(elements):
            current = elements[i].copy()
            text = current["text"].strip()
            
            # Проверяем, заканчивается ли текст на завершающий знак препинания
            sentence_endings = ('.', '!', '?', '…')
            
            # Склеиваем со следующими элементами, пока не встретим завершённое предложение
            while i + 1 < len(elements) and not text.endswith(sentence_endings):
                next_elem = elements[i + 1]
                
                # Проверяем, что следующий элемент не является заголовком
                if "header" in next_elem["label"].lower():
                    # Не склеиваем с заголовками
                    break
                
                # Склеиваем текст
                next_text = next_elem["text"].strip()
                # Добавляем пробел, если нужно
                if text and not text.endswith((' ', '-')):
                    text = text + " " + next_text
                else:
                    text = text + next_text
                
                # НЕ объединяем page_numbers! Оставляем только из первого элемента
                # Это предотвращает накопление page_numbers из всех склеенных фрагментов
                # current["page_numbers"] остаются без изменений
                
                # Headings обновляем, если в следующем элементе появились новые
                if next_elem["headings"] and next_elem["headings"] != current["headings"]:
                    current["headings"] = next_elem["headings"]
                
                i += 1
                
                logger.debug(f"Склеен фрагмент (page_numbers от первого): '{current['text'][:50]}...' + '{next_text[:50]}...'")
            
            current["text"] = text
            merged.append(current)
            i += 1

        logger.info(f"  Склеено элементов: {len(elements)} → {len(merged)}")
        
        return merged

    def chunk_document(self, doc: DoclingDocument) -> Iterator[SentenceChunk]:
        """
        Разбивает документ на чанки с сохранением метаданных.
        
        Чанки не пересекают границы заголовков - при смене любого заголовка
        (главы, аннотации и т.п.) текущий чанк автоматически завершается.

        Args:
            doc: DoclingDocument для чанкинга

        Yields:
            SentenceChunk объекты с информацией о чанках
        """
        logger.info("Начинаю чанкинг документа...")

        # Извлекаем элементы с метаданными
        elements = self._extract_elements_with_metadata(doc)
        logger.info(f"  Извлечено {len(elements)} элементов")

        # Собираем все предложения с их метаданными
        sentences_with_meta: list[dict[str, Any]] = []
        for elem in elements:
            elem_sentences = self.split_into_sentences(elem["text"])
            for sent in elem_sentences:
                sentences_with_meta.append({
                    "text": sent,
                    "page_numbers": elem["page_numbers"],
                    "headings": elem["headings"],
                    "filename": elem["filename"],
                })

        logger.info(f"  Разбито на {len(sentences_with_meta)} предложений")

        # Создаем чанки с агрегацией метаданных и учетом границ глав
        chunks = self._create_chunks_with_metadata(sentences_with_meta)
        logger.info(f"  Создано {len(chunks)} чанков")

        # Генерируем SentenceChunk объекты
        for i, chunk_data in enumerate(chunks, 1):
            yield SentenceChunk(
                chunk_id=f"chunk_{i}",
                text=chunk_data["text"],
                tokens=chunk_data["tokens"],
                sentence_count=chunk_data["sentence_count"],
                page_numbers=chunk_data["page_numbers"],
                headings=chunk_data["headings"],
                filename=chunk_data["filename"],
                metadata={
                    "chunking_method": "sentence_based",
                    "max_tokens": self.max_tokens,
                    "heading_aware": True,
                },
            )

        logger.info("Чанкинг завершен")

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
        chunks: Iterator[SentenceChunk],
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

        # Статистика
        if chunks_list:
            avg_tokens = sum(c.tokens for c in chunks_list) / len(chunks_list)
            min_tokens = min(c.tokens for c in chunks_list)
            max_tokens = max(c.tokens for c in chunks_list)
            logger.info("Статистика чанков:")
            logger.info(f"  Средний размер: {avg_tokens:.1f} токенов")
            logger.info(f"  Минимальный: {min_tokens} токенов")
            logger.info(f"  Максимальный: {max_tokens} токенов")

        return len(chunks_list)

