"""CLI для обработки документов: парсинг и чанкинг."""

import argparse
import logging
import sys
from pathlib import Path

from temporal_graph_rag import settings
from temporal_graph_rag.document_processing.chunker import (
    DocumentChunker,
    chunk_document_from_json,
)
from temporal_graph_rag.document_processing.pdf_parser import PDFParser
from temporal_graph_rag.document_processing.sentence_based_chunker import (
    SentenceBasedChunker,
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_pdf_command(args: argparse.Namespace) -> None:
    """
    Команда для парсинга PDF файла.

    Args:
        args: Аргументы командной строки
    """
    logger.info("=" * 60)
    logger.info("ПАРСИНГ PDF ДОКУМЕНТА")
    logger.info("=" * 60)

    parser = PDFParser(
        do_ocr=args.ocr,
        do_table_structure=args.tables,
        filter_toc=args.filter_toc,
        toc_max_page=args.toc_max_page,
    )

    try:
        doc = parser.parse_and_save(args.input, args.output)
        logger.info("✓ Парсинг успешно завершен")

        # Статистика
        if hasattr(doc, "pages") and doc.pages:
            logger.info(f"  Страниц: {len(doc.pages)}")

    except Exception as e:
        logger.error(f"✗ Ошибка при парсинге: {e}")
        sys.exit(1)


def chunk_document_command(args: argparse.Namespace) -> None:
    """
    Команда для чанкинга документа.

    Args:
        args: Аргументы командной строки
    """
    logger.info("=" * 60)
    logger.info("ЧАНКИНГ ДОКУМЕНТА")
    logger.info("=" * 60)
    logger.info(f"Метод чанкинга: {args.chunking_method}")

    try:
        if args.chunking_method == "sentence":
            # Используем sentence-based chunker
            from temporal_graph_rag.document_processing.chunker import load_document_from_json

            doc = load_document_from_json(args.input)
            chunker = SentenceBasedChunker(
                embed_model_id=args.model,
                max_tokens=args.max_tokens,
            )
            count = chunker.chunk_and_save(doc, args.output)
        else:
            # Используем hybrid chunker (docling)
            count = chunk_document_from_json(
                input_json=args.input,
                output_json=args.output,
                embed_model_id=args.model,
                max_tokens=args.max_tokens,
                merge_peers=args.merge_peers,
            )
        logger.info(f"✓ Чанкинг успешно завершен. Создано чанков: {count}")

    except Exception as e:
        logger.error(f"✗ Ошибка при чанкинге: {e}")
        sys.exit(1)


def process_pdf_full(args: argparse.Namespace) -> None:
    """
    Полная обработка PDF: парсинг + чанкинг.

    Args:
        args: Аргументы командной строки
    """
    logger.info("=" * 60)
    logger.info("ПОЛНАЯ ОБРАБОТКА PDF: ПАРСИНГ + ЧАНКИНГ")
    logger.info("=" * 60)

    # Определяем пути для промежуточных файлов
    pdf_path = Path(args.input)
    pdf_name = pdf_path.stem

    # Путь для промежуточного JSON (парсинг)
    if args.parsed_dir:
        parsed_dir = Path(args.parsed_dir)
    else:
        parsed_dir = Path("output/parsing_and_chunks/processed")

    parsed_json = parsed_dir / f"{pdf_name}.json"

    # Путь для чанков
    if args.chunks_dir:
        chunks_dir = Path(args.chunks_dir)
    else:
        chunks_dir = Path("output/parsing_and_chunks/chunks")

    chunks_json = chunks_dir / f"{pdf_name}_chunks.json"

    # Шаг 1: Парсинг
    logger.info("\n--- ШАГ 1: Парсинг PDF ---")
    parser = PDFParser(
        do_ocr=args.ocr,
        do_table_structure=args.tables,
        filter_toc=args.filter_toc,
        toc_max_page=args.toc_max_page,
    )

    try:
        doc = parser.parse_and_save(pdf_path, parsed_json)
        logger.info(f"✓ Парсинг завершен: {parsed_json}")

        if hasattr(doc, "pages") and doc.pages:
            logger.info(f"  Страниц: {len(doc.pages)}")

    except Exception as e:
        logger.error(f"✗ Ошибка при парсинге: {e}")
        sys.exit(1)

    # Шаг 2: Чанкинг
    logger.info("\n--- ШАГ 2: Чанкинг документа ---")
    logger.info(f"Метод чанкинга: {args.chunking_method}")

    try:
        if args.chunking_method == "sentence":
            chunker = SentenceBasedChunker(
                embed_model_id=args.model,
                max_tokens=args.max_tokens,
            )
        else:
            chunker = DocumentChunker(
                embed_model_id=args.model,
                max_tokens=args.max_tokens,
                merge_peers=args.merge_peers,
            )

        count = chunker.chunk_and_save(doc, chunks_json)
        logger.info(f"✓ Чанкинг завершен: {chunks_json}")
        logger.info(f"  Чанков: {count}")

    except Exception as e:
        logger.error(f"✗ Ошибка при чанкинге: {e}")
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("✓ ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО")
    logger.info("=" * 60)
    logger.info(f"Парсинг: {parsed_json}")
    logger.info(f"Чанки:   {chunks_json}")


def batch_process_pdfs(args: argparse.Namespace) -> None:
    """
    Пакетная обработка нескольких PDF файлов.

    Args:
        args: Аргументы командной строки
    """
    logger.info("=" * 60)
    logger.info("ПАКЕТНАЯ ОБРАБОТКА PDF ДОКУМЕНТОВ")
    logger.info("=" * 60)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"✗ Директория не найдена: {input_dir}")
        sys.exit(1)

    # Находим все PDF файлы
    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"✗ PDF файлы не найдены в: {input_dir}")
        sys.exit(1)

    logger.info(f"Найдено PDF файлов: {len(pdf_files)}")

    # Создаем парсер и чанкер один раз
    parser = PDFParser(
        do_ocr=args.ocr,
        do_table_structure=args.tables,
        filter_toc=args.filter_toc,
        toc_max_page=args.toc_max_page,
    )

    if args.chunking_method == "sentence":
        chunker = SentenceBasedChunker(
            embed_model_id=args.model,
            max_tokens=args.max_tokens,
        )
    else:
        chunker = DocumentChunker(
            embed_model_id=args.model,
            max_tokens=args.max_tokens,
            merge_peers=args.merge_peers,
        )

    # Директории для результатов
    parsed_dir = Path(args.parsed_dir) if args.parsed_dir else Path("output/parsing_and_chunks/processed")
    chunks_dir = Path(args.chunks_dir) if args.chunks_dir else Path("output/parsing_and_chunks/chunks")

    # Обрабатываем каждый файл
    success_count = 0
    failed_files = []

    for i, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"\n--- [{i}/{len(pdf_files)}] Обработка: {pdf_file.name} ---")

        pdf_name = pdf_file.stem
        parsed_json = parsed_dir / f"{pdf_name}.json"
        chunks_json = chunks_dir / f"{pdf_name}_chunks.json"

        try:
            # Парсинг
            logger.info("Парсинг...")
            doc = parser.parse_and_save(pdf_file, parsed_json)

            # Чанкинг
            logger.info("Чанкинг...")
            count = chunker.chunk_and_save(doc, chunks_json)

            logger.info(f"✓ Успешно: {count} чанков")
            success_count += 1

        except Exception as e:
            logger.error(f"✗ Ошибка при обработке {pdf_file.name}: {e}")
            failed_files.append(pdf_file.name)

    # Итоговая статистика
    logger.info("\n" + "=" * 60)
    logger.info("ИТОГОВАЯ СТАТИСТИКА")
    logger.info("=" * 60)
    logger.info(f"Всего файлов: {len(pdf_files)}")
    logger.info(f"Успешно: {success_count}")
    logger.info(f"Ошибок: {len(failed_files)}")

    if failed_files:
        logger.info("\nФайлы с ошибками:")
        for name in failed_files:
            logger.info(f"  - {name}")
        sys.exit(1)
    else:
        logger.info("\n✓ ВСЕ ФАЙЛЫ ОБРАБОТАНЫ УСПЕШНО")


def main() -> None:
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        description="Обработка документов: парсинг PDF и чанкинг",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Команда для выполнения")

    # Команда: parse
    parse_parser = subparsers.add_parser("parse", help="Парсинг PDF файла")
    parse_parser.add_argument("input", type=str, help="Путь к PDF файлу")
    parse_parser.add_argument("output", type=str, help="Путь для сохранения JSON")
    parse_parser.add_argument(
        "--ocr",
        action="store_true",
        default=settings.PDF_DO_OCR,
        help="Использовать OCR",
    )
    parse_parser.add_argument(
        "--no-tables",
        dest="tables",
        action="store_false",
        default=settings.PDF_DO_TABLE_STRUCTURE,
        help="Отключить обработку таблиц",
    )
    parse_parser.add_argument(
        "--no-filter-toc",
        dest="filter_toc",
        action="store_false",
        default=settings.PDF_FILTER_TABLE_OF_CONTENTS,
        help="Не фильтровать оглавление",
    )
    parse_parser.add_argument(
        "--toc-max-page",
        type=int,
        default=settings.PDF_TOC_MAX_PAGE,
        help="Макс. страница для поиска оглавления",
    )

    # Команда: chunk
    chunk_parser = subparsers.add_parser("chunk", help="Чанкинг документа")
    chunk_parser.add_argument("input", type=str, help="Путь к JSON с документом")
    chunk_parser.add_argument("output", type=str, help="Путь для сохранения чанков")
    chunk_parser.add_argument(
        "--model",
        type=str,
        default=settings.CHUNKING_EMBED_MODEL_ID,
        help="ID модели для токенизации",
    )
    chunk_parser.add_argument(
        "--max-tokens",
        type=int,
        default=settings.CHUNKING_MAX_TOKENS,
        help="Максимальное количество токенов в чанке",
    )
    chunk_parser.add_argument(
        "--chunking-method",
        type=str,
        choices=["hybrid", "sentence"],
        default=settings.CHUNKING_METHOD,
        help="Метод чанкинга: hybrid (Docling) или sentence (на основе предложений)",
    )
    chunk_parser.add_argument(
        "--no-merge-peers",
        dest="merge_peers",
        action="store_false",
        default=True,
        help="Не объединять соседние чанки (только для hybrid)",
    )

    # Команда: process (парсинг + чанкинг)
    process_parser = subparsers.add_parser(
        "process", help="Полная обработка: парсинг + чанкинг"
    )
    process_parser.add_argument("input", type=str, help="Путь к PDF файлу")
    process_parser.add_argument(
        "--parsed-dir",
        type=str,
        help="Директория для промежуточного JSON (по умолчанию: output/parsing_and_chunks/processed)",
    )
    process_parser.add_argument(
        "--chunks-dir",
        type=str,
        help="Директория для чанков (по умолчанию: output/parsing_and_chunks/chunks)",
    )
    process_parser.add_argument("--ocr", action="store_true", default=settings.PDF_DO_OCR)
    process_parser.add_argument(
        "--no-tables", dest="tables", action="store_false", default=settings.PDF_DO_TABLE_STRUCTURE
    )
    process_parser.add_argument(
        "--no-filter-toc", dest="filter_toc", action="store_false", default=settings.PDF_FILTER_TABLE_OF_CONTENTS
    )
    process_parser.add_argument(
        "--toc-max-page", type=int, default=settings.PDF_TOC_MAX_PAGE
    )
    process_parser.add_argument(
        "--model", type=str, default=settings.CHUNKING_EMBED_MODEL_ID
    )
    process_parser.add_argument(
        "--max-tokens", type=int, default=settings.CHUNKING_MAX_TOKENS
    )
    process_parser.add_argument(
        "--chunking-method",
        type=str,
        choices=["hybrid", "sentence"],
        default=settings.CHUNKING_METHOD,
        help="Метод чанкинга: hybrid (Docling) или sentence (на основе предложений)",
    )
    process_parser.add_argument(
        "--no-merge-peers", dest="merge_peers", action="store_false", default=True
    )

    # Команда: batch (пакетная обработка)
    batch_parser = subparsers.add_parser("batch", help="Пакетная обработка PDF файлов")
    batch_parser.add_argument("input_dir", type=str, help="Директория с PDF файлами")
    batch_parser.add_argument(
        "--parsed-dir",
        type=str,
        help="Директория для промежуточных JSON",
    )
    batch_parser.add_argument(
        "--chunks-dir",
        type=str,
        help="Директория для чанков",
    )
    batch_parser.add_argument("--ocr", action="store_true", default=settings.PDF_DO_OCR)
    batch_parser.add_argument(
        "--no-tables", dest="tables", action="store_false", default=settings.PDF_DO_TABLE_STRUCTURE
    )
    batch_parser.add_argument(
        "--no-filter-toc", dest="filter_toc", action="store_false", default=settings.PDF_FILTER_TABLE_OF_CONTENTS
    )
    batch_parser.add_argument(
        "--toc-max-page", type=int, default=settings.PDF_TOC_MAX_PAGE
    )
    batch_parser.add_argument(
        "--model", type=str, default=settings.CHUNKING_EMBED_MODEL_ID
    )
    batch_parser.add_argument(
        "--max-tokens", type=int, default=settings.CHUNKING_MAX_TOKENS
    )
    batch_parser.add_argument(
        "--chunking-method",
        type=str,
        choices=["hybrid", "sentence"],
        default=settings.CHUNKING_METHOD,
        help="Метод чанкинга: hybrid (Docling) или sentence (на основе предложений)",
    )
    batch_parser.add_argument(
        "--no-merge-peers", dest="merge_peers", action="store_false", default=True
    )

    args = parser.parse_args()

    if args.command == "parse":
        parse_pdf_command(args)
    elif args.command == "chunk":
        chunk_document_command(args)
    elif args.command == "process":
        process_pdf_full(args)
    elif args.command == "batch":
        batch_process_pdfs(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

