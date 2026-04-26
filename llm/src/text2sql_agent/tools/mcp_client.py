"""Module containing the Mock Model Context Protocol (MCP) Database Client."""

import sqlite3
import logging

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
            cursor = conn.cursor()
            
            # Extract DDLs
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = cursor.fetchall()
            
            schema_dump = []
            for table_name, create_sql in tables:
                schema_dump.append(f"----- Table: {table_name} -----")
                schema_dump.append(f"{create_sql};")
                
                # Fetch 3 sample rows
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                samples = cursor.fetchall()
                schema_dump.append(f"Sample Rows: {samples}\n")
                
            conn.close()
            return "\n".join(schema_dump)
        except Exception as e:
            logger.error(f"[MCPDatabaseClient] Failed to extract schema from {db_path}: {e}")
            return f"Error extracting schema: {e}"
