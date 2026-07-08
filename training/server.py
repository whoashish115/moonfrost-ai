"""
Web chat server for a locally-trained checkpoint.

Run it:
    python server.py                 # http://127.0.0.1:8000
    python server.py --port 8080 --checkpoint ../checkpoints/sft_last.pt

What this fixes compared to the previous version:

  * Checkpoints are loaded to CPU first and only the model weights are moved
    to the GPU. Training checkpoints carry AdamW's optimizer state, which is
    two thirds of the file -- the old `torch.load(path, map_location="cuda")`
    pushed all 4.7GB of sft_best.pt onto a 6GB card, where it either failed
    outright or spilled into shared system memory and ran at a crawl.

  * Generation runs on a worker thread. The old handler was `async def` but
    called straight into PyTorch, which blocks the event loop -- so while one
    reply was streaming, the server could not answer any other request,
    including the request to stop that very generation.

  * The stream sends JSON events with per-token deltas. The old one
    re-sent the entire reply on every token with newlines hand-escaped,
    which is quadratic in traffic and corrupts any output containing a
    literal backslash-n.

  * A reply can be stopped, regenerated, and every sampling parameter is
    settable per request.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import asyncio
import glob
import json
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone
from typing import Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

import checkpoint_utils
from chat_format import (DEFAULT_SAMPLING, ChatSpecialTokens, IncrementalTextDecoder,
                         build_prompt_token_ids)
from config import ModelConfig, count_active_parameters_per_token, count_total_parameters
from model import GPT

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIRECTORY = os.path.abspath(os.path.join(HERE, "..", "checkpoints"))
TOKENIZER_DIRECTORY = os.path.abspath(os.path.join(HERE, "..", "tokenizer"))
STATIC_DIRECTORY = os.path.join(HERE, "static")
DATABASE_PATH = os.path.join(HERE, "db", "chats.db")


# ----------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------

def open_database():
    """One short-lived connection per operation. SQLite connections are not
    safe to share across threads, and generation now runs on worker threads,
    so a single module-level connection (as before) would eventually corrupt
    or raise."""
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")  # lets a read happen while a write is in flight
    return connection


def migrate_database(connection):
    """Adds columns that were introduced after the first databases were created. Reading
    the table info is cheaper than a try/except around every ALTER, and it keeps an
    existing chat history rather than asking for a fresh database."""
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(sessions)")}
    if "archived" not in columns:
        connection.execute("ALTER TABLE sessions ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
        connection.commit()
        print("[db] added sessions.archived", flush=True)
    if "pinned" not in columns:
        connection.execute("ALTER TABLE sessions ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
        connection.commit()
        print("[db] added sessions.pinned", flush=True)
    if "workspace" not in columns:
        # every existing chat belongs to the default workspace, so nothing disappears
        connection.execute(
            "ALTER TABLE sessions ADD COLUMN workspace TEXT NOT NULL DEFAULT 'default'")
        connection.commit()
        print("[db] added sessions.workspace", flush=True)


def initialize_database():
    with closing(open_database()) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                model TEXT,
                token_count INTEGER,
                tokens_per_second REAL
            );
            CREATE INDEX IF NOT EXISTS messages_by_session ON messages(session_id, id);
            """
        )
        # tolerate databases created by the previous schema, which lacked the stats columns
        existing_columns = {row["name"] for row in connection.execute("PRAGMA table_info(messages)")}
        for column_name, column_type in (("model", "TEXT"), ("token_count", "INTEGER"),
                                          ("tokens_per_second", "REAL")):
            if column_name not in existing_columns:
                connection.execute(f"ALTER TABLE messages ADD COLUMN {column_name} {column_type}")
        connection.commit()
        migrate_database(connection)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------
# Model management
