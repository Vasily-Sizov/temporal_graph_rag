"""Модели данных для Temporal Graph RAG."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TemporalPosition:
    """
    Временная позиция текстовой единицы.
    
    Attributes:
        book_number: Номер книги (1-7 для Гарри Поттера)
        book_title: Название книги
        chunk_index: Индекс чанка внутри книги
        global_chunk_index: Глобальный индекс чанка (через все книги)
        relative_position: Относительная позиция в книге (0.0-1.0)
    """
    
    book_number: int
    book_title: str
    chunk_index: int
    global_chunk_index: int
    relative_position: float
    
    def __str__(self) -> str:
        """Строковое представление."""
        return (
            f"Книга {self.book_number}: {self.book_title}, "
            f"чанк {self.chunk_index} ({self.relative_position:.1%})"
        )


@dataclass
class TextUnit:
    """
    Текстовая единица (чанк) с временными метаданными.
    
    Attributes:
        id: Уникальный идентификатор
        text: Основной текст чанка
        contextualized_text: Контекстуализированный текст (с заголовками)
        temporal_position: Временная позиция
        text_tokens: Количество токенов в тексте
        metadata: Дополнительные метаданные
        entities: Список сущностей, извлеченных из этого чанка
        relationships: Список отношений, извлеченных из этого чанка
        embedding: Векторное представление (опционально)
    """
    
    id: str
    text: str
    contextualized_text: str
    temporal_position: TemporalPosition
    text_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)
    entities: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    embedding: list[float] | None = None


@dataclass
class Entity:
    """
    Сущность в графе знаний.
    
    Каждая запись = одно упоминание entity в конкретном контексте (книга + глава).
    
    Attributes:
        name: Имя сущности (нормализованное)
        type: Тип сущности (PERSON, LOCATION, EVENT и т.д.)
        book_number: Номер книги
        chapter: Глава к которой относится entity
        description: Описание сущности в этом контексте
        descriptions_raw: Список всех сырых описаний из чанков
        text_unit_ids: ID текстовых единиц, где упоминается сущность
        embedding: Векторное представление (опционально)
    """
    
    name: str
    type: str
    book_number: int
    chapter: str
    description: str = ""
    descriptions_raw: list[str] = field(default_factory=list)
    text_unit_ids: list[str] = field(default_factory=list)
    embedding: list[float] | None = None


@dataclass
class Relationship:
    """
    Отношение между сущностями.
    
    Каждая запись = одно упоминание relationship в конкретном контексте (книга + глава).
    
    Attributes:
        source: Исходная сущность
        target: Целевая сущность
        book_number: Номер книги
        chapter: Глава к которой относится отношение
        description: Описание отношения в этом контексте
        descriptions_raw: Список всех сырых описаний из чанков
        type: Тип отношения (OPERATIONAL, FACTUAL, NARRATIVE)
        weight: Вес отношения (0.0-1.0)
        text_unit_ids: ID текстовых единиц, где упоминается отношение
    """
    
    source: str
    target: str
    book_number: int
    chapter: str
    description: str = ""
    descriptions_raw: list[str] = field(default_factory=list)
    type: str = "UNKNOWN"
    weight: float = 0.5
    text_unit_ids: list[str] = field(default_factory=list)


@dataclass
class Community:
    """
    Временное сообщество (temporal window community).
    
    Attributes:
        id: Уникальный идентификатор
        book_number: Номер книги
        temporal_range: Диапазон чанков (start, end)
        relative_position_range: Относительная позиция в книге (start%, end%)
        title: Краткое название сообщества
        summary: Краткое резюме (1-2 предложения)
        report: Полный отчет о сообществе
        entities: Список имен сущностей в сообществе
        relationships: Список ID отношений в сообществе
        text_unit_ids: Список ID текстовых единиц
        embedding: Векторное представление отчета (опционально)
    """
    
    id: str
    book_number: int
    temporal_range: tuple[int, int]
    relative_position_range: tuple[float, float]
    title: str
    summary: str
    report: str
    entities: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    text_unit_ids: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    
    def contains_chunk(self, chunk_index: int) -> bool:
        """
        Проверяет, содержит ли сообщество чанк с данным индексом.
        
        Args:
            chunk_index: Индекс чанка
            
        Returns:
            True если чанк в диапазоне сообщества
        """
        start, end = self.temporal_range
        return start <= chunk_index <= end
    
    def overlaps_with(self, other: "Community") -> bool:
        """
        Проверяет, пересекается ли это сообщество с другим.
        
        Args:
            other: Другое сообщество
            
        Returns:
            True если сообщества пересекаются
        """
        if self.book_number != other.book_number:
            return False
        
        start1, end1 = self.temporal_range
        start2, end2 = other.temporal_range
        
        return not (end1 < start2 or end2 < start1)

