"""Форматирование результатов поиска в Markdown."""

from datetime import datetime
from pathlib import Path
from typing import Any

from temporal_graph_rag.models.data_models import Community, Entity, Relationship


def save_search_result_to_markdown(
    result: dict[str, Any],
    output_file: str | Path,
) -> None:
    """
    Сохраняет результат поиска в 2 Markdown файла:
    1. *_detailed_report_llm.md - полный контекст LLM в таблицах
    2. *_report_llm.md - только промпты (System + User)
    
    Args:
        result: Результат поиска из TemporalLocalSearch.search()
        output_file: Путь к выходному файлу (базовое имя)
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Создаем 2 файла
    base_name = output_path.stem
    detailed_report_path = output_path.parent / f"{base_name}_detailed_report_llm.md"
    report_path = output_path.parent / f"{base_name}_report_llm.md"
    
    # Сохраняем полный контекст в таблицах
    _save_detailed_report(result, detailed_report_path)
    
    # Сохраняем промпты
    _save_prompts_report(result, report_path)


def _save_detailed_report(result: dict[str, Any], output_path: Path) -> None:
    """Сохраняет отчет с ПОЛНЫМ контекстом LLM в табличном формате."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        # Заголовок
        f.write("# Temporal Graph RAG Query Result (TABULAR)\n\n")
        f.write(f"*Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("**Этот файл содержит ПОЛНЫЙ контекст LLM в табличном формате.**\n\n")
        
        # Вопрос
        f.write("## Вопрос\n\n")
        f.write(f"{result['query']}\n\n")
        
        # Фильтр по книгам
        if result.get("book_filter"):
            f.write(f"**Фильтр по книгам:** {', '.join(map(str, result['book_filter']))}\n\n")
        
        # Ответ
        f.write("## Ответ\n\n")
        f.write(f"{result['answer']}\n\n")
        
        # Статистика
        f.write("## Статистика контекста\n\n")
        f.write(f"- **Entities:** {len(result['entities_used'])}\n")
        f.write(f"- **Relationships:** {len(result['relationships_used'])}\n")
        f.write(f"- **Communities:** {len(result['communities_used'])}\n\n")
        
        # Контекст (ТОЧНО как в LLM, но в таблицах!)
        f.write("## Контекст отправленный в LLM (табличный формат)\n\n")
        
        # 1. Community Reports (таблица)
        f.write("### Community Reports\n\n")
        f.write("| # | Title | Summary |\n")
        f.write("|---|-------|----------|\n")
        for idx, comm in enumerate(result['communities_used'], 1):
            title = comm.title.replace("|", "\\|")
            summary = comm.summary.replace("|", "\\|").replace("\n", " ")
            f.write(f"| {idx} | {title} | {summary} |\n")
        f.write("\n")
        
        # 2. Entities + их Relationships (сгруппировано как в LLM!)
        f.write("### Entities\n\n")
        
        # Группируем relationships по entities (как в LLM)
        entity_text_units = {}
        for entity in result['entities_used']:
            entity_text_units[entity.name] = set(entity.text_unit_ids)
        
        for idx, entity in enumerate(result['entities_used'], 1):
            # Entity таблица
            f.write(f"#### Entity #{idx}: {entity.name}\n\n")
            f.write("| Attribute | Value |\n")
            f.write("|-----------|-------|\n")
            f.write(f"| Name | {entity.name} |\n")
            f.write(f"| Type | {entity.type} |\n")
            f.write(f"| Book | {entity.book_number} |\n")
            chapter = entity.chapter.replace("|", "\\|")
            f.write(f"| Chapter | {chapter} |\n")
            desc = entity.description.replace("|", "\\|").replace("\n", " ")
            f.write(f"| Description | {desc} |\n")
            f.write(f"| Text Units | {len(entity.text_unit_ids)} |\n")
            f.write("\n")
            
            # Relationships для этой entity (ТА ЖЕ ЛОГИКА, что в temporal_local_search.py!)
            # Берем relationships где source ИЛИ target совпадает по СОСТАВНОМУ КЛЮЧУ (name, book, chapter)
            # (включая out-network relationships, где второй конец НЕ в entities_used)
            entity_key = f"{entity.name}|{entity.book_number}|{entity.chapter}"
            entity_rels = [
                rel for rel in result['relationships_used']
                if (
                    f"{rel.source}|{rel.book_number}|{rel.chapter}" == entity_key
                    or f"{rel.target}|{rel.book_number}|{rel.chapter}" == entity_key
                )
            ]
            
            # Сортируем по (book_number, chapter) для хронологического порядка
            entity_rels.sort(key=lambda r: (r.book_number, r.chapter))
            
            if entity_rels:
                f.write(f"**Relationships для {entity.name}:**\n\n")
                f.write("| # | Source | Target | Description | Type | Weight | Book | Chapter | Text Units |\n")
                f.write("|---|--------|--------|-------------|------|--------|------|---------|------------|\n")
                for rel_idx, rel in enumerate(entity_rels, 1):
                    rel_desc = rel.description.replace("|", "\\|").replace("\n", " ")
                    rel_chapter = rel.chapter.replace("|", "\\|")
                    f.write(f"| {rel_idx} | {rel.source} | {rel.target} | {rel_desc} | {rel.type} | {rel.weight} | {rel.book_number} | {rel_chapter} | {len(rel.text_unit_ids)} |\n")
                f.write("\n")
        
        # 3. Sources (text units из LLM контекста с метаинформацией)
        f.write("### Sources\n\n")
        
        # Извлекаем source IDs из LLM контекста (в том порядке, в котором они были отправлены)
        import re
        sources_pattern = r'\*\*Source (book\d+_chunk_\d+)\*\*:'
        source_ids = re.findall(sources_pattern, result['context'])
        
        text_units_dict = result.get('text_units_dict', {})
        entities = result.get('entities_used', [])
        
        if source_ids and text_units_dict and entities:
            # Создаем mapping entity_name -> entity для быстрого доступа
            entities_dict = {e.name: e for e in entities}
            
            # Группируем sources по entities (в порядке из LLM контекста)
            current_entity = None
            source_idx = 0
            
            for source_id in source_ids:
                if source_id not in text_units_dict:
                    continue
                    
                text_unit = text_units_dict[source_id]
                
                # Находим entity для этого text_unit
                entity_for_source = None
                for entity in entities:
                    if source_id in entity.text_unit_ids:
                        entity_for_source = entity
                        break
                
                # Если сменилась entity, выводим заголовок
                if entity_for_source and entity_for_source != current_entity:
                    if source_idx > 0:
                        f.write("\n")
                    f.write(f"#### Sources для {entity_for_source.name}:\n\n")
                    current_entity = entity_for_source
                
                source_idx += 1
                f.write(f"**Source #{source_idx}: {text_unit.id}**\n\n")
                
                f.write("| Attribute | Value |\n")
                f.write("|-----------|-------|\n")
                f.write(f"| ID | {text_unit.id} |\n")
                f.write(f"| Book | {text_unit.temporal_position.book_number} |\n")
                f.write(f"| Book Title | {text_unit.temporal_position.book_title} |\n")
                chapter = text_unit.metadata.get('headings', ['Unknown'])[0] if text_unit.metadata.get('headings') else 'Unknown'
                f.write(f"| Chapter | {chapter} |\n")
                f.write(f"| Chunk Index | {text_unit.temporal_position.chunk_index} |\n")
                f.write(f"| Relative Position | {text_unit.temporal_position.relative_position:.2%} |\n")
                f.write(f"| Text Tokens | {text_unit.text_tokens} |\n")
                f.write("\n**Text:**\n\n")
                f.write(f"{text_unit.text.strip()}\n\n")
        else:
            f.write("*Sources не найдены.*\n\n")
        


def _save_prompts_report(result: dict[str, Any], output_path: Path) -> None:
    """Сохраняет отчет с промптами (System + User)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        # Заголовок
        f.write("# Temporal Graph RAG Query Result (LLM Prompts)\n\n")
        f.write(f"*Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("**Этот файл содержит промпты, отправленные в LLM.**\n\n")
        
        # Вопрос
        f.write("## Вопрос\n\n")
        f.write(f"{result['query']}\n\n")
        
        # Фильтр по книгам
        if result.get("book_filter"):
            f.write(f"**Фильтр по книгам:** {', '.join(map(str, result['book_filter']))}\n\n")
        
        # Ответ
        f.write("## Ответ LLM\n\n")
        f.write(f"{result['answer']}\n\n")
        
        # Статистика
        f.write("## Статистика контекста\n\n")
        f.write(f"- **Entities:** {len(result['entities_used'])}\n")
        f.write(f"- **Relationships:** {len(result['relationships_used'])}\n")
        f.write(f"- **Communities:** {len(result['communities_used'])}\n\n")
        
        # Промпты (без полного контекста)
        f.write("---\n\n")
        f.write("## Промпты\n\n")
        
        f.write("### System Prompt\n\n")
        f.write("```\n")
        f.write(result['system_prompt'])
        f.write("\n```\n\n")
        
        f.write("### User Prompt\n\n")
        f.write("```\n")
        f.write(result['user_prompt'])
        f.write("\n```\n\n")


def format_search_result_summary(result: dict[str, Any]) -> str:
    """
    Форматирует краткую сводку результата поиска для вывода в консоль.
    
    Args:
        result: Результат поиска
        
    Returns:
        Отформатированная строка
    """
    lines = []
    lines.append("=" * 80)
    lines.append("РЕЗУЛЬТАТ ПОИСКА")
    lines.append("=" * 80)
    lines.append(f"\nЗапрос: {result['query']}")
    
    if result.get("book_filter"):
        lines.append(f"Книги: {result['book_filter']}")
    
    lines.append(f"\nИспользовано:")
    lines.append(f"  - Entities: {len(result['entities_used'])}")
    lines.append(f"  - Relationships: {len(result['relationships_used'])}")
    lines.append(f"  - Communities: {len(result['communities_used'])}")
    lines.append("\n" + "-" * 80)
    lines.append("ОТВЕТ:")
    lines.append("-" * 80)
    lines.append(result["answer"])
    lines.append("=" * 80)
    
    return "\n".join(lines)

