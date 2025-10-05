import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from sqlalchemy.orm import Session

from ..database.models import FileAnalysis as DBAnalysis
from ..database.models import db


IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    "node_modules",
}


def fetch_records(limit: Optional[int] = None) -> list[DBAnalysis]:
    session: Session = db.get_session()
    try:
        query = session.query(DBAnalysis).order_by(DBAnalysis.file_name)
        if limit:
            query = query.limit(limit)
        return list(query)
    finally:
        session.close()


def filter_records(records: Iterable[DBAnalysis], include_ignored: bool) -> list[DBAnalysis]:
    if include_ignored:
        return list(records)

    filtered = []
    for record in records:
        parts = set(Path(record.file_path).parts)
        if parts.isdisjoint(IGNORED_DIRECTORIES):
            filtered.append(record)
    return filtered


def format_summary(text: Optional[str], max_length: int, full_summary: bool) -> str:
    summary = text or ""
    if full_summary or len(summary) <= max_length:
        return summary
    return summary[: max_length - 3].rstrip() + "..."


def resolve_agent(record: DBAnalysis) -> str:
    # Prefer extra_metadata -> agent, fall back to legacy metadata
    for metadata in (record.extra_metadata, record.metadata):
        if isinstance(metadata, dict) and metadata.get("agent"):
            return str(metadata["agent"])
    return "unknown"


def format_record(
    record: DBAnalysis,
    show_metadata: bool,
    summary_length: int,
    full_summary: bool,
) -> str:
    lines = [
        f"File: {record.file_name}",
        f"  Path: {record.file_path}",
        f"  Type: {record.file_type}",
        f"  Agent: {resolve_agent(record)}",
        f"  Summary: {format_summary(record.summary, summary_length, full_summary)}",
    ]

    tag_list = record.tags or []
    lines.append(f"  Tags: {', '.join(tag_list)}" if tag_list else "  Tags: -")

    if show_metadata:
        extra = json.dumps(record.extra_metadata or {}, indent=2, ensure_ascii=False)
        lines.append(f"  Extra Metadata: {extra}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Inspect stored file analyses")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of records displayed",
    )
    parser.add_argument(
        "--include-ignored",
        action="store_true",
        help="Display records from ignored directories such as .venv/",
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Include extra metadata in the output",
    )
    parser.add_argument(
        "--summary-length",
        type=int,
        default=180,
        help="Maximum length of the summary before truncation",
    )
    parser.add_argument(
        "--full-summary",
        action="store_true",
        help="Print the entire summary without truncation",
    )

    args = parser.parse_args()
    records = fetch_records(limit=args.limit)
    records = filter_records(records, include_ignored=args.include_ignored)

    if not records:
        print("No records found in the database.")
        return

    for record in records:
        print(
            format_record(
                record,
                show_metadata=args.show_metadata,
                summary_length=args.summary_length,
                full_summary=args.full_summary,
            )
        )
        print("-" * 60)


if __name__ == "__main__":
    main()
