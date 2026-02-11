"""Парсер PDF документов с улучшенной обработкой структуры."""

import logging
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument

from temporal_graph_rag import settings

logger = logging.getLogger(__name__)


class PDFParser:
    """Парсер PDF документов с фильтрацией оглавления и улучшенной обработкой глав."""

    def __init__(
        self,
        do_ocr: bool = settings.PDF_DO_OCR,
        do_table_structure: bool = settings.PDF_DO_TABLE_STRUCTURE,
        filter_toc: bool = settings.PDF_FILTER_TABLE_OF_CONTENTS,
        toc_max_page: int = settings.PDF_TOC_MAX_PAGE,
    ) -> None:
        """
        Инициализация парсера PDF.

        Args:
            do_ocr: Использовать ли OCR при парсинге
            do_table_structure: Обрабатывать ли структуру таблиц
            filter_toc: Фильтровать ли оглавление
            toc_max_page: Максимальная страница для поиска оглавления
        """
        self.do_ocr = do_ocr
        self.do_table_structure = do_table_structure
        self.filter_toc = filter_toc
        self.toc_max_page = toc_max_page

        logger.info(f"Инициализация PDFParser: OCR={do_ocr}, Tables={do_table_structure}")
        logger.info(f"Фильтрация оглавления: {filter_toc} (до страницы {toc_max_page})")

        # Настройка опций для PDF
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = do_ocr
        pipeline_options.do_table_structure = do_table_structure

        # Создание конвертера
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def parse(self, pdf_path: str | Path) -> DoclingDocument:
        """
        Парсит PDF файл и возвращает DoclingDocument.

        Args:
            pdf_path: Путь к PDF файлу

        Returns:
            DoclingDocument с обработанным содержимым

        Raises:
            FileNotFoundError: Если файл не найден
            RuntimeError: Если произошла ошибка при парсинге
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"Файл не найден: {pdf_path}")

        logger.info(f"Начинаю парсинг файла: {pdf_path}")

        try:
            # Конвертация документа
            result = self.converter.convert(str(pdf_path))
            doc = result.document

            # Логирование информации о документе
            if hasattr(doc, "pages") and doc.pages:
                logger.info(f"Документ обработан. Страниц: {len(doc.pages)}")
            else:
                logger.info("Документ обработан")

            # Фильтрация оглавления если требуется
            if self.filter_toc:
                doc = self._filter_table_of_contents(doc)

            # Исправление таблиц, которые на самом деле являются заголовками глав
            doc = self._fix_chapter_tables(doc)

            return doc

        except Exception as e:
            logger.error(f"Ошибка при парсинге PDF: {e}", exc_info=True)
            raise RuntimeError(f"Не удалось распарсить PDF: {e}") from e

    def parse_and_save(
        self,
        pdf_path: str | Path,
        output_path: str | Path,
    ) -> DoclingDocument:
        """
        Парсит PDF файл и сохраняет результат в JSON.

        Args:
            pdf_path: Путь к PDF файлу
            output_path: Путь для сохранения JSON файла

        Returns:
            DoclingDocument с обработанным содержимым
        """
        doc = self.parse(pdf_path)

        # Сохранение результата
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        json_data = doc.model_dump_json(indent=2)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json_data)

        logger.info(f"Результат сохранен в: {output_path}")
        return doc

    def _filter_table_of_contents(self, doc: DoclingDocument) -> DoclingDocument:
        """
        Фильтрует элементы оглавления из документа.

        Args:
            doc: Исходный документ

        Returns:
            Документ с отфильтрованным оглавлением
        """
        if not hasattr(doc, "body") or not hasattr(doc.body, "children"):
            return doc

        logger.info("Фильтрация оглавления...")

        # Подсчитываем удаленные элементы для статистики
        removed_count = 0
        filtered_children = []

        for child_ref in doc.body.children:
            # Получаем ссылку - child_ref это Pydantic модель с атрибутом cref
            cref = getattr(child_ref, "cref", None) if hasattr(child_ref, "cref") else child_ref.get("cref", "") if isinstance(child_ref, dict) else ""
            
            # Получаем элемент по ссылке
            item = self._resolve_reference(doc, cref)
            if item is None:
                filtered_children.append(child_ref)
                continue

            # Проверяем, является ли элемент частью оглавления
            if self._is_toc_item(item):
                removed_count += 1
                label = item.get('label', 'unknown') if isinstance(item, dict) else getattr(item, 'label', 'unknown')
                logger.debug(f"Удален элемент оглавления: {label}")
                continue

            filtered_children.append(child_ref)

        # Обновляем children
        doc.body.children = filtered_children

        logger.info(f"Удалено элементов оглавления: {removed_count}")
        return doc

    def _is_toc_item(self, item: dict[str, Any]) -> bool:
        """
        Проверяет, является ли элемент частью оглавления.

        Фильтрует элементы оглавления вида "Глава 1", "Глава 2", и т.д.,
        но НЕ трогает реальные заголовки глав вида "Глава 1 Мальчик, который выжил".

        Args:
            item: Элемент документа

        Returns:
            True если элемент является частью оглавления
        """
        # Проверяем метку элемента - оглавление это всегда list_item
        # Реальные главы - это section_header
        label = item.get("label", "")
        if label != "list_item":
            return False

        # Проверяем страницу
        prov = item.get("prov", [])
        if not prov:
            return False

        page_no = prov[0].get("page_no", float("inf"))
        if page_no > self.toc_max_page:
            return False

        # Проверяем текст - содержит ли "Глава" и только номер (без названия)
        text = item.get("text", "")
        if not text:
            return False

        text_stripped = text.strip()
        text_lower = text_stripped.lower()

        # Паттерны для оглавления: "Глава N" где N - это число
        # Длина должна быть короткой (обычно 7-10 символов для "Глава 12")
        # Реальные главы типа "Глава 1 Мальчик, который выжил" гораздо длиннее
        if text_lower.startswith("глава") and len(text_stripped) <= 12:
            # Дополнительная проверка: после "Глава" должен быть только номер
            # и ничего больше (или максимум точка/двоеточие)
            parts = text_stripped.split()
            if len(parts) <= 2:  # "Глава" + номер (или "Глава" + "1:")
                return True

        return False

    def _resolve_reference(
        self,
        doc: DoclingDocument,
        ref: str,
    ) -> dict[str, Any] | None:
        """
        Разрешает ссылку на элемент документа.

        Args:
            doc: Документ
            ref: Ссылка (например, "#/texts/0")

        Returns:
            Элемент документа или None
        """
        if not ref or not ref.startswith("#/"):
            return None

        parts = ref[2:].split("/")
        if len(parts) != 2:
            return None

        collection_name, index_str = parts
        try:
            index = int(index_str)
        except ValueError:
            return None

        # Получаем коллекцию
        collection = getattr(doc, collection_name, None)
        if collection is None or not isinstance(collection, list):
            return None

        # Получаем элемент
        if index < 0 or index >= len(collection):
            return None

        item = collection[index]
        # Конвертируем в dict если это Pydantic модель
        if hasattr(item, "model_dump"):
            return item.model_dump()
        return item if isinstance(item, dict) else None

    def _fix_chapter_tables(self, doc: DoclingDocument) -> DoclingDocument:
        """
        Исправляет таблицы, которые на самом деле являются заголовками глав.
        
        Docling иногда распознает заголовки глав как таблицы с ячейками типа:
        ["Глава N", "Название главы"]
        
        Args:
            doc: Исходный документ
            
        Returns:
            Документ с исправленными заголовками
        """
        if not hasattr(doc, "tables") or not doc.tables:
            return doc
            
        logger.info("Проверка таблиц на наличие заголовков глав...")
        
        import re
        
        fixed_count = 0
        for table in doc.tables:
            # Проверяем, есть ли data с table_cells
            if not hasattr(table, "data") or not table.data:
                continue
                
            table_data = table.data
            if hasattr(table_data, "model_dump"):
                table_data = table_data.model_dump()
            elif not isinstance(table_data, dict):
                continue
                
            cells = table_data.get("table_cells", [])
            if not cells or len(cells) > 3:  # Заголовок главы обычно 1-2 ячейки
                continue
            
            # Собираем текст из ячеек
            cell_texts = []
            for cell in cells:
                if isinstance(cell, dict):
                    text = cell.get("text", "").strip()
                else:
                    text = getattr(cell, "text", "").strip() if hasattr(cell, "text") else ""
                if text:
                    cell_texts.append(text)
            
            if not cell_texts:
                continue
            
            # Проверяем, похоже ли это на заголовок главы
            combined_text = " ".join(cell_texts)
            
            # Паттерны для глав: "Глава N", "Глава N ..." и т.д.
            chapter_pattern = re.compile(
                r"^Глава\s+\d+",
                re.IGNORECASE | re.UNICODE
            )
            
            if chapter_pattern.match(combined_text):
                # Это заголовок главы! Просто меняем label
                logger.info(f"Исправлена таблица на заголовок: {combined_text}")
                
                # Меняем label на section_header
                # Это позволит HybridChunker правильно обработать как заголовок
                table.label = "section_header"
                
                fixed_count += 1
        
        logger.info(f"Исправлено таблиц-заголовков: {fixed_count}")
        return doc


def parse_pdf_file(
    pdf_path: str | Path,
    output_path: str | Path | None = None,
    do_ocr: bool = settings.PDF_DO_OCR,
    do_table_structure: bool = settings.PDF_DO_TABLE_STRUCTURE,
    filter_toc: bool = settings.PDF_FILTER_TABLE_OF_CONTENTS,
) -> DoclingDocument:
    """
    Удобная функция для парсинга одного PDF файла.

    Args:
        pdf_path: Путь к PDF файлу
        output_path: Путь для сохранения (опционально)
        do_ocr: Использовать ли OCR
        do_table_structure: Обрабатывать ли таблицы
        filter_toc: Фильтровать ли оглавление

    Returns:
        DoclingDocument с обработанным содержимым
    """
    parser = PDFParser(
        do_ocr=do_ocr,
        do_table_structure=do_table_structure,
        filter_toc=filter_toc,
    )

    if output_path:
        return parser.parse_and_save(pdf_path, output_path)
    else:
        return parser.parse(pdf_path)

