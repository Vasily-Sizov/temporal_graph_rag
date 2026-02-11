"""Загрузчик и обогащение чанков временными метаданными."""

import json
import logging
import re
from pathlib import Path
from typing import Any

from temporal_graph_rag.models.data_models import TemporalPosition, TextUnit

logger = logging.getLogger(__name__)


class ChunkLoader:
    """Загрузчик чанков с обогащением временными метаданными."""

    @staticmethod
    def extract_book_number(filename: str) -> int:
        """
        Извлекает номер книги из имени файла.

        Args:
            filename: Имя файла

        Returns:
            Номер книги (1-7)

        Raises:
            ValueError: Если не удалось извлечь номер книги
        """
        # Паттерн для извлечения номера из начала имени файла
        match = re.match(r"^(\d+)", filename)
        if match:
            return int(match.group(1))

        raise ValueError(f"Не удалось извлечь номер книги из: {filename}")

    @staticmethod
    def extract_book_title(filename: str) -> str:
        """
        Извлекает название книги из имени файла.

        Args:
            filename: Имя файла

        Returns:
            Название книги
        """
        # Убираем расширение и номер
        name = Path(filename).stem
        # Убираем префикс с номером (например, "01_")
        name = re.sub(r"^\d+[_\s-]*", "", name)
        # Заменяем подчеркивания и дефисы на пробелы
        name = name.replace("_", " ").replace("-", " ")
        # Убираем "_chunks" в конце если есть
        name = re.sub(r"\s*chunks\s*$", "", name, flags=re.IGNORECASE)
        return name.strip()

    @staticmethod
    def load_chunks_from_file(
        file_path: str | Path,
    ) -> list[TextUnit]:
        """
        Загружает чанки из JSON файла и обогащает их временными метаданными.

        Args:
            file_path: Путь к JSON файлу с чанками

        Returns:
            Список TextUnit объектов

        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если не удалось распарсить файл
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        logger.info(f"Загрузка чанков из: {file_path}")

        # Извлекаем метаданные из имени файла
        book_number = ChunkLoader.extract_book_number(file_path.name)
        book_title = ChunkLoader.extract_book_title(file_path.name)

        # Загружаем JSON
        with open(file_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        if not isinstance(chunks_data, list):
            raise ValueError(f"Ожидался список чанков, получено: {type(chunks_data)}")

        # Конвертируем в TextUnit объекты
        text_units = []
        total_chunks = len(chunks_data)

        for idx, chunk_data in enumerate(chunks_data):
            chunk_id = chunk_data.get("chunk_id", f"chunk_{idx + 1}")
            text = chunk_data.get("text", "")
            contextualized_text = chunk_data.get("contextualized_text", text)
            text_tokens = chunk_data.get("text_tokens", 0)
            metadata = chunk_data.get("metadata", {})
            
            # Сохраняем headings в metadata (информация о главе)
            headings = chunk_data.get("headings", [])
            if headings:
                metadata["headings"] = headings

            # Создаем временную позицию
            temporal_position = TemporalPosition(
                book_number=book_number,
                book_title=book_title,
                chunk_index=idx,
                global_chunk_index=-1,  # Будет установлен позже при загрузке всех книг
                relative_position=idx / max(total_chunks - 1, 1),
            )

            # Создаем TextUnit
            text_unit = TextUnit(
                id=f"book{book_number}_{chunk_id}",
                text=text,
                contextualized_text=contextualized_text,
                temporal_position=temporal_position,
                text_tokens=text_tokens,
                metadata=metadata,
            )

            text_units.append(text_unit)

        logger.info(
            f"Загружено {len(text_units)} чанков из книги {book_number}: {book_title}"
        )
        return text_units

    @staticmethod
    def load_chunks_from_directory(
        directory_path: str | Path,
    ) -> list[TextUnit]:
        """
        Загружает чанки из всех JSON файлов в директории.

        Args:
            directory_path: Путь к директории с JSON файлами

        Returns:
            Список TextUnit объектов из всех книг

        Raises:
            FileNotFoundError: Если директория не найдена
        """
        directory_path = Path(directory_path)
        if not directory_path.exists():
            raise FileNotFoundError(f"Директория не найдена: {directory_path}")

        logger.info(f"Загрузка чанков из директории: {directory_path}")

        # Находим все JSON файлы
        json_files = sorted(directory_path.glob("*.json"))
        if not json_files:
            raise ValueError(f"Не найдено JSON файлов в: {directory_path}")

        logger.info(f"Найдено {len(json_files)} файлов")

        # Загружаем чанки из каждого файла
        all_text_units = []
        global_chunk_index = 0

        for json_file in json_files:
            text_units = ChunkLoader.load_chunks_from_file(json_file)

            # Устанавливаем глобальные индексы
            for text_unit in text_units:
                text_unit.temporal_position.global_chunk_index = global_chunk_index
                global_chunk_index += 1

            all_text_units.extend(text_units)

        logger.info(f"Всего загружено {len(all_text_units)} чанков из {len(json_files)} книг")
        return all_text_units

    @staticmethod
    def group_by_book(text_units: list[TextUnit]) -> dict[int, list[TextUnit]]:
        """
        Группирует текстовые единицы по номеру книги.

        Args:
            text_units: Список текстовых единиц

        Returns:
            Словарь {book_number: [text_units]}
        """
        grouped: dict[int, list[TextUnit]] = {}
        for text_unit in text_units:
            book_number = text_unit.temporal_position.book_number
            if book_number not in grouped:
                grouped[book_number] = []
            grouped[book_number].append(text_unit)

        return grouped

