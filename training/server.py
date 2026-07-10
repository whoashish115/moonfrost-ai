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
# ----------------------------------------------------------------------

class ModelManager:
    """Owns the single loaded model and serializes access to it.

    One GPU, one model: two generations running at once would both allocate
    KV caches and activations, which on a 6GB card is how you turn a working
    server into an out-of-memory error. The lock makes a second request wait
    rather than fail."""

    def __init__(self, device: str):
        self.device = device
        self.model: Optional[GPT] = None
        self.model_config: Optional[ModelConfig] = None
        self.checkpoint_name: Optional[str] = None
        self.checkpoint_info: dict = {}
        self.load_error: Optional[str] = None
        self.is_loading = False
        self.generation_lock = threading.Lock()
        self._load_lock = threading.Lock()

        self.tokenizer = Tokenizer.from_file(os.path.join(TOKENIZER_DIRECTORY, "tokenizer.json"))
        self.special_tokens = ChatSpecialTokens(self.tokenizer)

    def load(self, checkpoint_name: str) -> bool:
        path = os.path.join(CHECKPOINT_DIRECTORY, checkpoint_name)
        if not os.path.isfile(path):
            self.load_error = f"checkpoint not found: {checkpoint_name}"
            return False

        with self._load_lock:
            self.is_loading = True
            self.load_error = None
            try:
                # free the previous model before allocating the next one, or peak memory is
                # the sum of both
                if self.model is not None:
                    self.model = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()

                print(f"loading {checkpoint_name} -> {self.device} ...")
                started = time.time()
                model, model_config, info = checkpoint_utils.load_for_inference(
                    path, self.device, ModelConfig, GPT
                )
                model.enable_inference_absorption()

                if model_config.vocabulary_size != self.tokenizer.get_vocab_size():
                    print(f"  warning: checkpoint vocabulary_size={model_config.vocabulary_size} but the "
                          f"tokenizer has {self.tokenizer.get_vocab_size()} tokens; output will be garbled "
                          f"if these came from different tokenizer trainings")

                self.model, self.model_config, self.checkpoint_info = model, model_config, info
                self.checkpoint_name = checkpoint_name
                print(f"  loaded in {time.time() - started:.1f}s: "
                      f"{count_total_parameters(model_config)/1e6:.1f}M parameters, "
                      f"context {model_config.max_sequence_length}, step {info['step']}, "
                      f"val loss {info['best_val']:.4f}")
                return True
            except Exception as error:
                self.model = None
                self.load_error = f"{type(error).__name__}: {error}"
                print(f"  failed to load {checkpoint_name}: {self.load_error}")
                return False
            finally:
                self.is_loading = False

    def status(self):
        return {
            "loaded": self.model is not None,
            "loading": self.is_loading,
            "error": self.load_error,
            "checkpoint": self.checkpoint_name,
            "device": self.device,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "context_length": self.model_config.max_sequence_length if self.model_config else None,
            "total_parameters": count_total_parameters(self.model_config) if self.model_config else None,
            "active_parameters": count_active_parameters_per_token(self.model_config) if self.model_config else None,
            "step": self.checkpoint_info.get("step"),
            "best_val": self.checkpoint_info.get("best_val"),
            "vocabulary_size": self.tokenizer.get_vocab_size(),
        }


def choose_default_checkpoint() -> Optional[str]:
    """Prefer an instruction-tuned checkpoint (those answer questions);
    among those prefer an already-stripped inference-only file, then the
    smallest, since the smallest is the one most likely to fit."""
    candidates = checkpoint_utils.list_checkpoints(CHECKPOINT_DIRECTORY)
    candidates = [entry for entry in candidates if "error" not in entry]
    if not candidates:
        return None
    chat_tuned = [entry for entry in candidates if entry["is_chat_tuned"]] or candidates
    # Best validation loss wins. Sorting by size, as this once did, is arbitrary when two
    # checkpoints of the same architecture are byte-for-byte the same length -- which is
    # exactly the case with v0.1 and v1.0, so the server picked between them at random.
    def rank(entry):
        value = entry.get("best_val")
        return (entry["has_optimizer_state"],
                value if isinstance(value, (int, float)) and value == value else float("inf"))
    chat_tuned.sort(key=rank)
    return chat_tuned[0]["name"]


# ----------------------------------------------------------------------
# Request/response models
# ----------------------------------------------------------------------

class SamplingSettings(BaseModel):
    temperature: float = Field(DEFAULT_SAMPLING["temperature"], ge=0.0, le=2.0)
    top_k: int = Field(DEFAULT_SAMPLING["top_k"], ge=0, le=1000)
    top_p: float = Field(DEFAULT_SAMPLING["top_p"], ge=0.0, le=1.0)
    min_p: float = Field(DEFAULT_SAMPLING["min_p"], ge=0.0, le=1.0)
    repetition_penalty: float = Field(DEFAULT_SAMPLING["repetition_penalty"], ge=1.0, le=2.0)
    frequency_penalty: float = Field(DEFAULT_SAMPLING["frequency_penalty"], ge=0.0, le=2.0)
    presence_penalty: float = Field(DEFAULT_SAMPLING["presence_penalty"], ge=0.0, le=2.0)
    max_new_tokens: int = Field(DEFAULT_SAMPLING["max_new_tokens"], ge=1, le=4096)
    seed: Optional[int] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str
    system_prompt: Optional[str] = None
    settings: SamplingSettings = SamplingSettings()


class RegenerateRequest(BaseModel):
    session_id: str
    system_prompt: Optional[str] = None
    settings: SamplingSettings = SamplingSettings()


class LoadModelRequest(BaseModel):
    model_name: str


class RenameRequest(BaseModel):
    title: str


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------

app = FastAPI(title="Local LLM chat")
manager: Optional[ModelManager] = None
active_generations: Dict[str, threading.Event] = {}
active_generations_lock = threading.Lock()


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIRECTORY, "index.html"))
