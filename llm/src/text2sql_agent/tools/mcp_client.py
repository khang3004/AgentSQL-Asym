"""Module containing the database schema context client."""

import csv
import os
import sqlite3
import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

class MCPDatabaseClient:
    """
    A simulated Model Context Protocol (MCP) client designed to interact with SQLite.
    Provides methods to securely extract database metadata for Language Models.
    """

    def get_relevant_schema(self, query: str, db_path: str) -> str:
        """
        Simulates an MCP interaction to fetch pertinent schema DDL and sample rows.
        
        Args:
            query (str): The user's natural language query (for relevant filtering simulation).
            db_path (str): The filesystem path to the target database.
            
        Returns:
            str: A formatted string concatenating CREATE TABLE definitions and 3 sample rows.
            
        Raises:
            Exception: Wraps standard exceptions to protect the execution graph.
        """
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            descriptions = self._load_column_descriptions(db_path)

            cursor.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            tables = cursor.fetchall()

            schema_dump = []
            for table in tables:
                table_name = table["name"]
                create_sql = table["sql"]
                schema_dump.append(f"----- Table: {table_name} -----")
                schema_dump.append(f"{create_sql};")

                cursor.execute(f'PRAGMA table_info("{table_name}")')
                columns = cursor.fetchall()
                column_lines = []
                for col in columns:
                    col_name = col["name"]
                    desc = descriptions.get(table_name.lower(), {}).get(col_name.lower(), "")
                    desc_suffix = f" -- {desc}" if desc else ""
                    pk_suffix = " primary_key" if col["pk"] else ""
                    column_lines.append(
                        f"- {col_name} ({col['type'] or 'UNKNOWN'}{pk_suffix}){desc_suffix}"
                    )
                schema_dump.append("Columns:\n" + "\n".join(column_lines))

                cursor.execute(f'PRAGMA foreign_key_list("{table_name}")')
                fks = cursor.fetchall()
                if fks:
                    fk_lines = [
                        f"- {table_name}.{fk['from']} -> {fk['table']}.{fk['to']}"
                        for fk in fks
                    ]
                    schema_dump.append("Foreign keys:\n" + "\n".join(fk_lines))

                cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 3')
                samples = [dict(row) for row in cursor.fetchall()]
                schema_dump.append(f"Sample rows: {samples}\n")

            conn.close()
            return "\n".join(schema_dump)
        except Exception as e:
            logger.error(f"[MCPDatabaseClient] Failed to extract schema from {db_path}: {e}")
            return f"Error extracting schema: {e}"

    def _load_column_descriptions(self, db_path: str) -> dict[str, dict[str, str]]:
        """Load BIRD-style database_description CSV files when available."""
        desc_dir = os.path.join(os.path.dirname(db_path), "database_description")
        descriptions: dict[str, dict[str, str]] = defaultdict(dict)
        if not os.path.isdir(desc_dir):
            return descriptions

        for filename in os.listdir(desc_dir):
            if not filename.endswith(".csv"):
                continue
            table_name = filename[:-4].lower()
            path = os.path.join(desc_dir, filename)
            try:
                with open(path, newline="", encoding="latin-1") as fh:
                    reader = csv.reader(fh)
                    header = next(reader, [])
                    for row in reader:
                        parsed = self._parse_description_row(header, row)
                        if not parsed:
                            continue
                        column, description = parsed
                        if description:
                            descriptions[table_name][column.lower()] = description[:500]
            except Exception as exc:
                logger.debug("[MCPDatabaseClient] Skipped description file %s: %s", path, exc)
        return descriptions

    def _parse_description_row(self, header: list[str], row: list[str]) -> tuple[str, str] | None:
        if not row:
            return None
        normalized = [h.strip().lstrip("\ufeff").lower() for h in header]

        def by_name(names: tuple[str, ...], default_idx: int) -> Any:
            for name in names:
                if name in normalized:
                    idx = normalized.index(name)
                    if idx < len(row) and row[idx].strip():
                        return row[idx].strip()
            return row[default_idx].strip() if default_idx < len(row) else ""

        column = by_name(("original_column_name", "column_name", "column"), 0)
        description = by_name(("column_description", "description", "column_meaning"), 2)
        value_description = by_name(("value_description", "value_meaning"), 4)
        pieces = [piece for piece in [description, value_description] if piece]
        return (column, "; ".join(pieces)) if column else None
