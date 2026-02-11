"""Модуль для работы с Parquet хранилищем."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from temporal_graph_rag.models.data_models import (
    Community,
    Entity,
    Relationship,
    TemporalPosition,
    TextUnit,
)
from temporal_graph_rag.utils.id_utils import create_relationship_id

logger = logging.getLogger(__name__)


class ParquetStorage:
    """
    Класс для сохранения и загрузки данных в формате Parquet.

    Структура:
    - entities.parquet: данные сущностей без эмбеддингов
    - relationships.parquet: данные отношений
    - communities.parquet: данные communities без эмбеддингов
    - text_units.parquet: данные текстовых единиц
    """

    def __init__(self, base_path: str | Path) -> None:
        """
        Инициализация хранилища.

        Args:
            base_path: Базовый путь для хранения данных
        """
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "data"
        self.data_path.mkdir(parents=True, exist_ok=True)

    def save_entities(self, entities_dict: dict[str, Entity]) -> None:
        """
        Сохраняет entities в Parquet.

        Args:
            entities_dict: Словарь сущностей {name: Entity}
        """
        logger.info(f"Сохранение {len(entities_dict)} entities в Parquet...")

        records = []
        for entity in entities_dict.values():
            record = {
                "name": entity.name,
                "type": entity.type,
                "book_number": entity.book_number,
                "chapter": entity.chapter,
                "description": entity.description,
                "descriptions_raw": json.dumps(
                    entity.descriptions_raw, ensure_ascii=False
                ),
                "text_unit_ids": json.dumps(entity.text_unit_ids, ensure_ascii=False),
            }
            records.append(record)

        df = pd.DataFrame(records)
        output_file = self.data_path / "entities.parquet"
        df.to_parquet(output_file, index=False, compression="snappy")
        logger.info(f"Entities сохранены: {output_file}")

    def save_relationships(self, relationships_dict: dict[str, Relationship]) -> None:
        """
        Сохраняет relationships в Parquet.

        Args:
            relationships_dict: Словарь отношений {id: Relationship}
        """
        logger.info(f"Сохранение {len(relationships_dict)} relationships в Parquet...")

        records = []
        for rel in relationships_dict.values():
            # ID не сохраняем - будем генерировать при загрузке из source/target/book/chapter
            record = {
                "source": rel.source,
                "target": rel.target,
                "book_number": rel.book_number,
                "chapter": rel.chapter,
                "description": rel.description,
                "descriptions_raw": json.dumps(
                    rel.descriptions_raw, ensure_ascii=False
                ),
                "type": rel.type,
                "weight": rel.weight,
                "text_unit_ids": json.dumps(rel.text_unit_ids, ensure_ascii=False),
            }
            records.append(record)

        df = pd.DataFrame(records)
        output_file = self.data_path / "relationships.parquet"
        df.to_parquet(output_file, index=False, compression="snappy")
        logger.info(f"Relationships сохранены: {output_file}")

    def save_communities(self, communities: list[Community]) -> None:
        """
        Сохраняет communities в Parquet.

        Args:
            communities: Список communities
        """
        logger.info(f"Сохранение {len(communities)} communities в Parquet...")

        records = []
        for comm in communities:
            record = {
                "id": comm.id,
                "book_number": comm.book_number,
                "title": comm.title,
                "summary": comm.summary,
                "report": comm.report,
                "temporal_range_start": comm.temporal_range[0],
                "temporal_range_end": comm.temporal_range[1],
                "relative_position_start": comm.relative_position_range[0],
                "relative_position_end": comm.relative_position_range[1],
                "entities": json.dumps(comm.entities, ensure_ascii=False),
                "relationships": json.dumps(comm.relationships, ensure_ascii=False),
            }
            records.append(record)

        df = pd.DataFrame(records)
        output_file = self.data_path / "communities.parquet"
        df.to_parquet(output_file, index=False, compression="snappy")
        logger.info(f"Communities сохранены: {output_file}")

    def save_text_units(self, text_units: list[TextUnit]) -> None:
        """
        Сохраняет text_units в Parquet.

        Args:
            text_units: Список текстовых единиц
        """
        logger.info(f"Сохранение {len(text_units)} text_units в Parquet...")

        records = []
        for unit in text_units:
            # Не сохраняем contextualized_text - будем использовать text при загрузке
            record = {
                "id": unit.id,
                "text": unit.text,
                "book_number": unit.temporal_position.book_number,
                "chunk_index": unit.temporal_position.chunk_index,
                "temporal_position": json.dumps(
                    {
                        "book_number": unit.temporal_position.book_number,
                        "book_title": unit.temporal_position.book_title,
                        "chunk_index": unit.temporal_position.chunk_index,
                        "global_chunk_index": unit.temporal_position.global_chunk_index,
                        "relative_position": unit.temporal_position.relative_position,
                    },
                    ensure_ascii=False,
                ),
                "entities": json.dumps(unit.entities, ensure_ascii=False),
                "relationships": json.dumps(unit.relationships, ensure_ascii=False),
                "metadata": json.dumps(unit.metadata or {}, ensure_ascii=False),
            }
            records.append(record)

        df = pd.DataFrame(records)
        output_file = self.data_path / "text_units.parquet"
        df.to_parquet(output_file, index=False, compression="snappy")
        logger.info(f"Text units сохранены: {output_file}")

    def load_entities(self) -> dict[str, Entity]:
        """
        Загружает entities из Parquet.

        Returns:
            Словарь сущностей {name: Entity}
        """
        input_file = self.data_path / "entities.parquet"
        logger.info(f"Загрузка entities из {input_file}...")

        df = pd.read_parquet(input_file)
        entities_dict = {}

        for _, row in df.iterrows():
            entity = Entity(
                name=row["name"],
                type=row["type"],
                book_number=row["book_number"],
                chapter=row["chapter"],
                description=row["description"],
                descriptions_raw=json.loads(row.get("descriptions_raw", "[]")),
                text_unit_ids=json.loads(row["text_unit_ids"]),
                embedding=None,  # Эмбеддинги загружаются отдельно через VectorStorage
            )
            # Ключ = (name, book_number, chapter) для уникальности
            entity_key = f"{entity.name}|{entity.book_number}|{entity.chapter}"
            entities_dict[entity_key] = entity

        logger.info(f"Загружено {len(entities_dict)} entities")
        return entities_dict

    def load_relationships(self) -> dict[str, Relationship]:
        """
        Загружает relationships из Parquet.

        Returns:
            Словарь отношений {id: Relationship}
        """
        input_file = self.data_path / "relationships.parquet"
        logger.info(f"Загрузка relationships из {input_file}...")

        df = pd.read_parquet(input_file)
        relationships_dict = {}

        for _, row in df.iterrows():
            rel = Relationship(
                source=row["source"],
                target=row["target"],
                book_number=row["book_number"],
                chapter=row["chapter"],
                description=row["description"],
                descriptions_raw=json.loads(row.get("descriptions_raw", "[]")),
                type=row["type"],
                weight=row["weight"],
                text_unit_ids=json.loads(row["text_unit_ids"]),
            )
            # Генерируем MD5 ID как в GraphRAG: hash(source|target|book|chapter)
            rel_id = create_relationship_id(
                row["source"], row["target"], row["book_number"], row["chapter"]
            )
            relationships_dict[rel_id] = rel

        logger.info(f"Загружено {len(relationships_dict)} relationships")
        return relationships_dict

    def load_communities(self) -> list[Community]:
        """
        Загружает communities из Parquet.

        Returns:
            Список communities
        """
        input_file = self.data_path / "communities.parquet"
        logger.info(f"Загрузка communities из {input_file}...")

        df = pd.read_parquet(input_file)
        communities = []

        for _, row in df.iterrows():
            comm = Community(
                id=row["id"],
                book_number=row["book_number"],
                title=row["title"],
                summary=row["summary"],
                report=row["report"],
                temporal_range=(row["temporal_range_start"], row["temporal_range_end"]),
                relative_position_range=(
                    row["relative_position_start"],
                    row["relative_position_end"],
                ),
                entities=json.loads(row["entities"]),
                relationships=json.loads(row["relationships"]),
                embedding=None,  # Эмбеддинги загружаются отдельно через VectorStorage
            )
            communities.append(comm)

        logger.info(f"Загружено {len(communities)} communities")
        return communities

    def load_text_units(self) -> list[TextUnit]:
        """
        Загружает text_units из Parquet.

        Returns:
            Список текстовых единиц
        """
        input_file = self.data_path / "text_units.parquet"
        logger.info(f"Загрузка text_units из {input_file}...")

        df = pd.read_parquet(input_file)
        text_units = []

        for _, row in df.iterrows():
            temp_pos_data = json.loads(row["temporal_position"])
            temporal_position = TemporalPosition(
                book_number=temp_pos_data["book_number"],
                book_title=temp_pos_data["book_title"],
                chunk_index=temp_pos_data["chunk_index"],
                global_chunk_index=temp_pos_data["global_chunk_index"],
                relative_position=temp_pos_data["relative_position"],
            )

            unit = TextUnit(
                id=row["id"],
                text=row["text"],
                # Просто копируем text (contextualized_text не используется)
                contextualized_text=row["text"],
                temporal_position=temporal_position,
                text_tokens=0,  # Не сохраняем, можно пересчитать
                metadata=json.loads(row["metadata"]),
                entities=json.loads(row["entities"]),
                relationships=json.loads(row["relationships"]),
                embedding=None,
            )
            text_units.append(unit)

        logger.info(f"Загружено {len(text_units)} text_units")
        return text_units
