#!/usr/bin/env python3
"""
SQLite Database Creation Script
Creates a SQLite database with the specified schema for Chainlit data layer.

The schema includes tables for:
- users: User information with metadata
- threads: Conversation threads
- steps: Individual steps/messages in threads
- elements: File attachments and media elements
- feedbacks: User feedback on messages

Note: SQLite adaptations made:
- UUID fields use TEXT type (SQLite doesn't have native UUID)
- JSONB becomes TEXT (will store JSON as text)
- TEXT[] arrays become TEXT (will store as comma-separated or JSON)
"""

import sqlite3
import os
import sys
from datetime import datetime


def create_database(db_path: str = "chainlit.db"):
    """
    Create SQLite database with the specified schema.

    Args:
        db_path (str): Path where the database file will be created
    """

    # Check if database already exists
    if os.path.exists(db_path):
        response = input(f"Database '{db_path}' already exists. Overwrite? (y/N): ")
        if response.lower() != "y":
            print("Database creation cancelled.")
            return False
        os.remove(db_path)

    try:
        # Connect to SQLite database (creates file if it doesn't exist)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print(f"Creating database: {db_path}")

        # Enable foreign key constraints
        cursor.execute("PRAGMA foreign_keys = ON;")

        # Create users table
        print("Creating users table...")
        cursor.execute(
            """
            CREATE TABLE users (
                "id" TEXT PRIMARY KEY,
                "identifier" TEXT NOT NULL UNIQUE,
                "metadata" TEXT NOT NULL,
                "createdAt" TEXT
            );
        """
        )

        # Create threads table
        print("Creating threads table...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                "id" TEXT PRIMARY KEY,
                "createdAt" TEXT,
                "name" TEXT,
                "userId" TEXT,
                "userIdentifier" TEXT,
                "tags" TEXT,
                "metadata" TEXT,
                FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
            );
        """
        )

        # Create steps table
        print("Creating steps table...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS steps (
                "id" TEXT PRIMARY KEY,
                "name" TEXT NOT NULL,
                "type" TEXT NOT NULL,
                "threadId" TEXT NOT NULL,
                "parentId" TEXT,
                "streaming" BOOLEAN NOT NULL,
                "waitForAnswer" BOOLEAN,
                "isError" BOOLEAN,
                "metadata" TEXT,
                "tags" TEXT,
                "input" TEXT,
                "output" TEXT,
                "createdAt" TEXT,
                "command" TEXT,
                "start" TEXT,
                "end" TEXT,
                "generation" TEXT,
                "showInput" TEXT,
                "language" TEXT,
                "indent" INTEGER,
                "defaultOpen" BOOLEAN,
                FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
            );
        """
        )

        # Create elements table
        print("Creating elements table...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS elements (
                "id" TEXT PRIMARY KEY,
                "threadId" TEXT,
                "type" TEXT,
                "url" TEXT,
                "chainlitKey" TEXT,
                "name" TEXT NOT NULL,
                "display" TEXT,
                "objectKey" TEXT,
                "size" TEXT,
                "page" INTEGER,
                "language" TEXT,
                "forId" TEXT,
                "mime" TEXT,
                "props" TEXT,
                FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
            );
        """
        )

        # Create feedbacks table
        print("Creating feedbacks table...")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feedbacks (
                "id" TEXT PRIMARY KEY,
                "forId" TEXT NOT NULL,
                "threadId" TEXT NOT NULL,
                "value" INTEGER NOT NULL,
                "comment" TEXT,
                FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
            );
        """
        )

        # Create indexes for better performance
        print("Creating indexes...")

        # Index on user identifier for quick lookups
        cursor.execute('CREATE INDEX idx_users_identifier ON users("identifier");')

        # Indexes on foreign keys
        cursor.execute('CREATE INDEX idx_threads_userId ON threads("userId");')
        cursor.execute('CREATE INDEX idx_steps_threadId ON steps("threadId");')
        cursor.execute('CREATE INDEX idx_steps_parentId ON steps("parentId");')
        cursor.execute('CREATE INDEX idx_elements_threadId ON elements("threadId");')
        cursor.execute('CREATE INDEX idx_elements_forId ON elements("forId");')
        cursor.execute('CREATE INDEX idx_feedbacks_threadId ON feedbacks("threadId");')
        cursor.execute('CREATE INDEX idx_feedbacks_forId ON feedbacks("forId");')

        # Indexes on commonly queried fields
        cursor.execute('CREATE INDEX idx_threads_createdAt ON threads("createdAt");')
        cursor.execute('CREATE INDEX idx_steps_createdAt ON steps("createdAt");')
        cursor.execute('CREATE INDEX idx_steps_type ON steps("type");')

        # Commit changes
        conn.commit()

        # Verify tables were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        print(f"\nDatabase created successfully!")
        print(f"Location: {os.path.abspath(db_path)}")
        print(f"Tables created: {[table[0] for table in tables]}")

        # Show table info
        print("\nTable schemas:")
        for table_name in ["users", "threads", "steps", "elements", "feedbacks"]:
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            print(f"\n{table_name}:")
            for col in columns:
                print(f"  - {col[1]} {col[2]} {'(PK)' if col[5] else ''}")

        return True

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return False

    except Exception as e:
        print(f"Error creating database: {e}")
        return False

    finally:
        if conn:
            conn.close()


def main():
    """Main function to handle command line arguments and create database."""

    # Default database name
    db_name = "chainlit.db"

    # Check if custom database name provided
    if len(sys.argv) > 1:
        db_name = sys.argv[1]

    print("SQLite Database Creation Script")
    print("=" * 40)
    print(f"Target database: {db_name}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    success = create_database(db_name)

    if success:
        print(f"\n✅ Database '{db_name}' created successfully!")
        print(f"\nYou can now use this database with your Chainlit application.")
        print(
            f"Make sure to update your Chainlit configuration to use: {os.path.abspath(db_name)}"
        )
    else:
        print(f"\n❌ Failed to create database '{db_name}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
