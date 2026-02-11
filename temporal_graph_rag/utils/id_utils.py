"""Утилиты для генерации детерминированных ID (как в GraphRAG)."""

import hashlib


def create_entity_id(name: str, chapter: str) -> str:
    """
    Создает детерминированный MD5 ID для entity.
    
    Args:
        name: Имя сущности
        chapter: Глава
    
    Returns:
        MD5 хеш (32 символа)
    """
    content = f"{name}|{chapter}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def create_relationship_id(source: str, target: str, book_number: int | str, chapter: str) -> str:
    """
    Создает детерминированный MD5 ID для relationship.
    
    Args:
        source: Имя source entity
        target: Имя target entity
        book_number: Номер книги
        chapter: Глава
    
    Returns:
        MD5 хеш (32 символа)
    """
    content = f"{source}|{target}|{book_number}|{chapter}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()

