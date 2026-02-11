"""CLI для Temporal Graph RAG."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from temporal_graph_rag.indexing.index_builder import IndexBuilder
from temporal_graph_rag.search.index_loader import IndexLoader
from temporal_graph_rag.search.temporal_local_search import TemporalLocalSearch
from temporal_graph_rag.search.classic_rag import ClassicRAG
from temporal_graph_rag.settings import (
    CLASSIC_RAG_TOP_K_SEARCH,
    CLASSIC_RAG_TOP_K_RERANK,
    CLASSIC_RAG_USE_RERANKER,
)
from temporal_graph_rag.utils.api_client import APIClient
from temporal_graph_rag.utils.result_formatter import (
    save_search_result_to_markdown,
    format_search_result_summary,
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def build_index_command_async(args: argparse.Namespace) -> None:
    """
    Асинхронная команда для построения индекса.

    Args:
        args: Аргументы командной строки
    """
    logger.info("=" * 80)
    logger.info("ПОСТРОЕНИЕ ИНДЕКСА TEMPORAL GRAPH RAG (АСИНХРОННО)")
    logger.info("=" * 80)

    # Создаем API клиент
    api_client = APIClient()

    # Создаем построитель индекса
    index_builder = IndexBuilder(
        api_client=api_client,
        window_size=args.community_window_size,
        overlap=args.community_overlap,
    )

    # Строим индекс асинхронно
    try:
        stats = await index_builder.build_index_async(
            chunks_dir=args.chunks_dir,
            output_dir=args.output_dir,
        )

        logger.info("\nИндекс успешно построен!")
        logger.info(f"Статистика: {stats}")

    except Exception as e:
        logger.error(f"Ошибка построения индекса: {e}", exc_info=True)
        sys.exit(1)


def build_index_command(args: argparse.Namespace) -> None:
    """
    Команда для построения индекса (обертка для асинхронной версии).

    Args:
        args: Аргументы командной строки
    """
    asyncio.run(build_index_command_async(args))


def search_command(args: argparse.Namespace) -> None:
    """
    Команда для поиска.

    Args:
        args: Аргументы командной строки
    """
    logger.info("=" * 80)
    logger.info("TEMPORAL LOCAL SEARCH")
    logger.info("=" * 80)

    # Создаем API клиент
    api_client = APIClient()

    # Загружаем индекс
    logger.info(f"Загрузка индекса из: {args.index_dir}")
    try:
        index_loader = IndexLoader(args.index_dir)
        text_units, entities_dict, relationships_dict, communities = (
            index_loader.load_index()
        )
    except Exception as e:
        logger.error(f"Ошибка загрузки индекса: {e}", exc_info=True)
        sys.exit(1)

    # Создаем поисковик
    searcher = TemporalLocalSearch(
        api_client=api_client,
        index_loader=index_loader,
        text_units=text_units,
        entities_dict=entities_dict,
        relationships_dict=relationships_dict,
        communities=communities,
        top_k_entities=args.top_k_entities,
        top_k_communities=args.top_k,
    )

    # Парсим фильтр по книгам
    book_filter = None
    if args.books:
        try:
            book_filter = [int(b) for b in args.books.split(",")]
        except ValueError:
            logger.error(f"Неверный формат фильтра книг: {args.books}")
            sys.exit(1)

    # Выполняем поиск
    logger.info(f"\nЗапрос: {args.query}")
    try:
        result = searcher.search(
            query=args.query,
            book_filter=book_filter,
        )

        # Выводим результат в консоль
        print("\n" + format_search_result_summary(result))
        
        # Сохраняем в файл если указан
        if args.output:
            save_search_result_to_markdown(result, args.output)
            logger.info(f"\nРезультат сохранен в: {args.output}")

    except Exception as e:
        logger.error(f"Ошибка поиска: {e}", exc_info=True)
        sys.exit(1)


def classic_search_command(args: argparse.Namespace) -> None:
    """
    Команда для классического RAG поиска.

    Args:
        args: Аргументы командной строки
    """
    logger.info("=" * 80)
    logger.info("CLASSIC RAG SEARCH")
    logger.info("=" * 80)

    # Создаем API клиент
    api_client = APIClient()

    # Создаем классический RAG
    logger.info(f"Инициализация Classic RAG с индексом: {args.index_dir}")
    try:
        classic_rag = ClassicRAG(
            index_dir=args.index_dir,
            api_client=api_client,
            top_k_search=args.top_k_search,
            top_k_rerank=args.top_k_rerank,
            use_reranker=args.use_reranker,
        )
    except Exception as e:
        logger.error(f"Ошибка инициализации Classic RAG: {e}", exc_info=True)
        sys.exit(1)

    # Выполняем поиск
    logger.info(f"\nЗапрос: {args.query}")
    try:
        result = classic_rag.query(args.query)

        # Выводим результат в консоль
        print("\n" + "=" * 80)
        print("ОТВЕТ:")
        print("=" * 80)
        print(result["answer"])
        print("\n" + "=" * 80)
        print(f"ИСПОЛЬЗОВАННЫЕ ДОКУМЕНТЫ ({len(result['context_chunks'])}):")
        print("=" * 80)
        
        for i, chunk_data in enumerate(result["context_chunks"], 1):
            chunk = chunk_data["chunk"]
            chunk_id = chunk.get("chunk_id", chunk.get("id", "unknown"))
            book_number = chunk.get("book_number", "?")
            chapter = chunk.get("chapter", "?")
            rerank_score = chunk_data.get("rerank_score")
            faiss_score = chunk_data.get("faiss_score", 0)
            
            print(f"\n[{i}] ID: {chunk_id} (Книга {book_number}, Глава {chapter})")
            print(f"    FAISS Score: {faiss_score:.4f}")
            if rerank_score is not None:
                print(f"    Rerank Score: {rerank_score:.4f}")
            text_preview = chunk.get("text", "")[:200]
            print(f"    Text: {text_preview}...")
        
        # Сохраняем в файл если указан
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# Classic RAG Search Results\n\n")
                f.write(f"**Query:** {result['query']}\n\n")
                f.write(f"## Answer\n\n{result['answer']}\n\n")
                f.write(f"## Context Documents ({len(result['context_chunks'])})\n\n")
                
                for i, chunk_data in enumerate(result["context_chunks"], 1):
                    chunk = chunk_data["chunk"]
                    chunk_id = chunk.get("chunk_id", chunk.get("id", "unknown"))
                    book_number = chunk.get("book_number", "?")
                    chapter = chunk.get("chapter", "?")
                    rerank_score = chunk_data.get("rerank_score")
                    faiss_score = chunk_data.get("faiss_score", 0)
                    text = chunk.get("text", "")
                    
                    f.write(f"### Document {i}\n\n")
                    f.write(f"- **ID:** {chunk_id}\n")
                    f.write(f"- **Location:** Книга {book_number}, Глава {chapter}\n")
                    f.write(f"- **FAISS Score:** {faiss_score:.4f}\n")
                    if rerank_score is not None:
                        f.write(f"- **Rerank Score:** {rerank_score:.4f}\n")
                    f.write(f"\n**Text:**\n\n{text}\n\n")
                
                f.write(f"## System Prompt\n\n```\n{result['system_prompt']}\n```\n\n")
                f.write(f"## User Prompt\n\n```\n{result['user_prompt']}\n```\n")
            
            logger.info(f"\nРезультат сохранен в: {args.output}")

    except Exception as e:
        logger.error(f"Ошибка поиска: {e}", exc_info=True)
        sys.exit(1)


def main() -> None:
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        description="Temporal Graph RAG - временной граф знаний для серий книг"
    )

    subparsers = parser.add_subparsers(dest="command", help="Команды")

    # Команда build
    build_parser = subparsers.add_parser("build", help="Построить индекс из чанков")
    build_parser.add_argument(
        "chunks_dir",
        type=str,
        help="Директория с JSON файлами чанков",
    )
    build_parser.add_argument(
        "output_dir",
        type=str,
        help="Директория для сохранения индекса",
    )
    build_parser.add_argument(
        "--community-window-size",
        type=int,
        default=20,
        help="Размер временного окна для communities в чанках (по умолчанию: 20)",
    )
    build_parser.add_argument(
        "--community-overlap",
        type=int,
        default=5,
        help="Перекрытие между community окнами в чанках (по умолчанию: 5)",
    )

    # Команда search
    search_parser = subparsers.add_parser("search", help="Выполнить поиск по индексу")
    search_parser.add_argument(
        "index_dir",
        type=str,
        help="Директория с индексом",
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Поисковый запрос",
    )
    search_parser.add_argument(
        "--books",
        type=str,
        help="Фильтр по книгам (например: 1,2,3)",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Количество communities для контекста (по умолчанию: 5)",
    )
    search_parser.add_argument(
        "--top-k-entities",
        type=int,
        default=10,
        help="Количество entities для поиска (по умолчанию: 10)",
    )
    search_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Путь к файлу для сохранения детального результата в Markdown",
    )

    # Команда classic_search
    classic_search_parser = subparsers.add_parser(
        "classic_search", 
        help="Выполнить классический RAG поиск"
    )
    classic_search_parser.add_argument(
        "index_dir",
        type=str,
        help="Директория с индексом (должна содержать text_units.faiss и text_units.json)",
    )
    classic_search_parser.add_argument(
        "query",
        type=str,
        help="Поисковый запрос",
    )
    classic_search_parser.add_argument(
        "--top-k-search",
        type=int,
        default=CLASSIC_RAG_TOP_K_SEARCH,
        help=f"Количество документов для векторного поиска (по умолчанию: {CLASSIC_RAG_TOP_K_SEARCH})",
    )
    classic_search_parser.add_argument(
        "--top-k-rerank",
        type=int,
        default=CLASSIC_RAG_TOP_K_RERANK,
        help=f"Количество документов после reranking (по умолчанию: {CLASSIC_RAG_TOP_K_RERANK})",
    )
    classic_search_parser.add_argument(
        "--use-reranker",
        action="store_true",
        default=CLASSIC_RAG_USE_RERANKER,
        help="Использовать reranker для переранжирования",
    )
    classic_search_parser.add_argument(
        "--no-reranker",
        action="store_false",
        dest="use_reranker",
        help="Не использовать reranker",
    )
    classic_search_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Путь к файлу для сохранения результата в Markdown",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "build":
        build_index_command(args)
    elif args.command == "search":
        search_command(args)
    elif args.command == "classic_search":
        classic_search_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
