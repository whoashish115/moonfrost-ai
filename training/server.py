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


@app.get("/home")
async def home():
    """The project page. It lives in docs/ so the same file can be published as a site."""
    return FileResponse(os.path.join(HERE, "..", "docs", "index.html"))


@app.get("/api/status")
async def api_status():
    status = manager.status()
    if torch.cuda.is_available():
        status["vram_allocated_bytes"] = torch.cuda.memory_allocated()
        status["vram_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
    return status


@app.post("/api/sessions/archive_all")
async def api_archive_all(workspace: str = "default"):
    """Archives every active chat at once. Nothing is deleted, so this is reversible from
    the archive view."""
    with closing(open_database()) as connection:
        cursor = connection.execute(
            "UPDATE sessions SET archived = 1 WHERE archived = 0 AND workspace = ?", (workspace,))
        connection.commit()
    return {"ok": True, "archived": cursor.rowcount}


@app.delete("/api/sessions")
async def api_delete_all(archived_only: bool = False, workspace: str = "default"):
    """Deletes chats permanently. archived_only empties the archive and leaves active
    chats alone, which is the safer half of this and worth having separately."""
    with closing(open_database()) as connection:
        if archived_only:
            rows = connection.execute(
                "SELECT id FROM sessions WHERE archived = 1 AND workspace = ?", (workspace,)).fetchall()
        else:
            rows = connection.execute(
                "SELECT id FROM sessions WHERE workspace = ?", (workspace,)).fetchall()
        ids = [row["id"] for row in rows]
        for session_id in ids:
            connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        connection.commit()
    return {"ok": True, "deleted": len(ids)}


@app.get("/api/sessions/export")
async def api_export_sessions():
    """Every chat and message as one JSON document, for keeping or moving elsewhere."""
    with closing(open_database()) as connection:
        sessions = [dict(row) for row in connection.execute(
            "SELECT id, title, created_at, updated_at, archived FROM sessions ORDER BY created_at")]
        for session in sessions:
            session["messages"] = [dict(row) for row in connection.execute(
                """SELECT role, content, timestamp, model, token_count, tokens_per_second
                   FROM messages WHERE session_id = ? ORDER BY id""", (session["id"],))]
    return {"exported_at": now_iso(), "chats": len(sessions),
            "messages": sum(len(s["messages"]) for s in sessions), "sessions": sessions}


@app.get("/api/stats")
async def api_stats(workspace: str = "default"):
    """Counts for the data screen, so it can say what it is about to act on."""
    with closing(open_database()) as connection:
        row = connection.execute(
            """SELECT (SELECT COUNT(*) FROM sessions WHERE archived = 0 AND workspace = ?) AS active,
                      (SELECT COUNT(*) FROM sessions WHERE archived = 1 AND workspace = ?) AS archived,
                      (SELECT COUNT(*) FROM messages m JOIN sessions s ON s.id = m.session_id
                       WHERE s.workspace = ?) AS messages""",
            (workspace, workspace, workspace)).fetchone()
    return dict(row)


@app.get("/api/workspaces")
async def api_workspaces():
    """Which workspaces exist, and how much is in each. A workspace is just a label on a
    chat: separate lists that share one model and one database, which is all that is
    needed to keep unrelated conversations apart."""
    with closing(open_database()) as connection:
        rows = connection.execute(
            """SELECT workspace AS name, COUNT(*) AS chats,
                      SUM(CASE WHEN archived = 1 THEN 1 ELSE 0 END) AS archived
               FROM sessions GROUP BY workspace ORDER BY workspace""").fetchall()
    found = [dict(row) for row in rows]
    if not any(entry["name"] == "default" for entry in found):
        found.insert(0, {"name": "default", "chats": 0, "archived": 0})
    return {"workspaces": found}


@app.get("/api/models")
async def api_models():
    entries = checkpoint_utils.list_checkpoints(CHECKPOINT_DIRECTORY)
    return {"models": entries, "current": manager.checkpoint_name}


@app.post("/api/load_model")
async def api_load_model(request: LoadModelRequest):
    # loading a multi-gigabyte checkpoint takes seconds to minutes; keep it off the event loop
    ok = await asyncio.to_thread(manager.load, request.model_name)
    if not ok:
        return JSONResponse({"status": "error", "message": manager.load_error}, status_code=400)
    return {"status": "success", "model": request.model_name, "info": manager.status()}


@app.get("/api/sessions")
async def api_sessions(archived: bool = False, workspace: str = "default"):
    """Archived chats are hidden from the list rather than deleted, so a conversation can
    be put away without losing it."""
    with closing(open_database()) as connection:
        rows = connection.execute(
            # Newest first by creation, not by last reply: sorting on updated_at made an
            # old conversation jump over a newer one the moment you answered in it, so the
            # list reordered itself under the pointer.
            """SELECT s.id, s.title, s.created_at, s.updated_at, s.archived, s.pinned,
                      (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count
               FROM sessions s WHERE s.archived = ? AND s.workspace = ?
               ORDER BY s.pinned DESC, s.created_at DESC""",
            (1 if archived else 0, workspace)
        ).fetchall()
    return [dict(row) for row in rows]


class ArchiveRequest(BaseModel):
    archived: bool = True


class PinRequest(BaseModel):
    pinned: bool = True


@app.post("/api/sessions/{session_id}/pin")
async def api_pin_session(session_id: str, request: PinRequest):
    """Pinned chats sort to the top of their space. It is only an ordering flag, so a
    pinned chat behaves like any other in every other respect."""
    with closing(open_database()) as connection:
        cursor = connection.execute("UPDATE sessions SET pinned = ? WHERE id = ?",
                                    (1 if request.pinned else 0, session_id))
        connection.commit()
    if not cursor.rowcount:
        raise HTTPException(status_code=404, detail="no such chat")
    return {"ok": True, "pinned": request.pinned}


@app.post("/api/sessions/{session_id}/archive")
async def api_archive_session(session_id: str, request: ArchiveRequest):
    with closing(open_database()) as connection:
        cursor = connection.execute("UPDATE sessions SET archived = ? WHERE id = ?",
                                    (1 if request.archived else 0, session_id))
        connection.commit()
    if not cursor.rowcount:
        raise HTTPException(status_code=404, detail="no such chat")
    return {"ok": True, "archived": request.archived}


@app.post("/api/sessions")
async def api_create_session(workspace: str = "default"):
    session_id = str(uuid.uuid4())
    timestamp = now_iso()
    with closing(open_database()) as connection:
        connection.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at, workspace) VALUES (?, ?, ?, ?, ?)",
            (session_id, "New chat", timestamp, timestamp, workspace),
        )
        connection.commit()
    return {"id": session_id, "title": "New chat", "created_at": timestamp, "updated_at": timestamp,
            "message_count": 0}


@app.get("/api/sessions/{session_id}")
async def api_get_session(session_id: str):
    with closing(open_database()) as connection:
        session = connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        rows = connection.execute(
            """SELECT id, role, content, timestamp, model, token_count, tokens_per_second
               FROM messages WHERE session_id = ? ORDER BY id ASC""",
            (session_id,),
        ).fetchall()
    return {"session": dict(session), "messages": [dict(row) for row in rows]}


@app.patch("/api/sessions/{session_id}")
async def api_rename_session(session_id: str, request: RenameRequest):
    title = request.title.strip()[:120] or "New chat"
    with closing(open_database()) as connection:
        connection.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                            (title, now_iso(), session_id))
        connection.commit()
    return {"status": "success", "title": title}


@app.delete("/api/sessions/{session_id}")
async def api_delete_session(session_id: str):
    with closing(open_database()) as connection:
        connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        connection.commit()
    return {"status": "success"}


@app.delete("/api/messages/{message_id}")
async def api_delete_message(message_id: int):
    """Deletes a message and everything after it in the same session --
    conversation history is a prefix, so keeping later turns after removing
    an earlier one would leave the model reading a conversation that never
    happened."""
    with closing(open_database()) as connection:
        row = connection.execute("SELECT session_id FROM messages WHERE id = ?", (message_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="message not found")
        connection.execute("DELETE FROM messages WHERE session_id = ? AND id >= ?",
                            (row["session_id"], message_id))
        connection.commit()
    return {"status": "success"}


@app.post("/api/stop/{request_id}")
async def api_stop(request_id: str):
    with active_generations_lock:
        event = active_generations.get(request_id)
    if event is None:
        return {"status": "not_found"}
    event.set()
    return {"status": "stopping"}


def load_history(session_id: str) -> List[tuple]:
    with closing(open_database()) as connection:
        rows = connection.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,)
        ).fetchall()
    return [(row["role"], row["content"]) for row in rows]


def sse(event_name: str, payload: dict) -> str:
    """One server-sent event. The payload is JSON, so newlines, quotes and
    backslashes in model output survive the trip intact -- the previous
    hand-rolled '\\n' escaping did not."""
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def stream_reply(session_id: str, user_message: Optional[str],
                        system_prompt: Optional[str], settings: SamplingSettings,
                        http_request: Request):
    """Shared body of /api/chat and /api/regenerate.

    user_message is None for a regenerate, where the last user turn is
    already in the database and the assistant turn after it has been
    removed."""
    if manager.model is None:
        yield sse("error", {"message": manager.load_error or "no model is loaded"})
        return

    history = load_history(session_id)
    if user_message is None:
        # regenerate: the newest turn in the database is the user message to answer
        if not history or history[-1][0] != "user":
            yield sse("error", {"message": "nothing to regenerate"})
            return
        user_message = history[-1][1]
        history = history[:-1]
    else:
        timestamp = now_iso()
        with closing(open_database()) as connection:
            connection.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, "user", user_message, timestamp),
            )
            if not history:
                title = user_message.strip().split("\n")[0][:60] or "New chat"
                connection.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                                    (title, timestamp, session_id))
            else:
                connection.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (timestamp, session_id))
            connection.commit()

    special = manager.special_tokens
    context_length = manager.model_config.max_sequence_length
    # leave room for the reply: the prompt may use at most the context minus the reply length
    max_prompt_tokens = max(16, context_length - settings.max_new_tokens)
    prompt_token_ids = build_prompt_token_ids(
        manager.tokenizer, history, user_message, special,
        system_prompt=system_prompt, max_prompt_tokens=max_prompt_tokens,
    )

    request_id = str(uuid.uuid4())
    stop_event = threading.Event()
    with active_generations_lock:
        active_generations[request_id] = stop_event

    yield sse("start", {"request_id": request_id, "prompt_tokens": len(prompt_token_ids),
                        "model": manager.checkpoint_name})

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    def generate_on_worker_thread():
        """Runs the blocking PyTorch generation loop off the event loop and
        hands each text delta back through the asyncio queue."""
        decoder = IncrementalTextDecoder(manager.tokenizer, skip_special_tokens=True)
        token_count = 0
        started = time.time()
        try:
            with manager.generation_lock:
                prompt_tensor = torch.tensor([prompt_token_ids], dtype=torch.long, device=manager.device)
                for token_id in manager.model.generate_stream(
                    prompt_tensor,
                    max_new_tokens=settings.max_new_tokens,
                    temperature=settings.temperature,
                    top_k=settings.top_k or None,
                    top_p=settings.top_p if settings.top_p < 1.0 else None,
                    min_p=settings.min_p or None,
                    stop_token_ids=special.stop_ids,
                    repetition_penalty=settings.repetition_penalty,
                    frequency_penalty=settings.frequency_penalty,
                    presence_penalty=settings.presence_penalty,
                    repetition_window=DEFAULT_SAMPLING["repetition_window"],
                    protected_token_ids=special.all_special_ids,
                    seed=settings.seed,
                    should_stop=stop_event.is_set,
                ):
                    if token_id in special.stop_ids:
                        break
                    token_count += 1
                    delta = decoder.push(token_id)
                    if delta:
                        loop.call_soon_threadsafe(queue.put_nowait, {"delta": delta})
                # release any character still held back mid-encoding at the cut-off point
                trailing = decoder.flush()
                if trailing:
                    loop.call_soon_threadsafe(queue.put_nowait, {"delta": trailing})
            elapsed = max(time.time() - started, 1e-6)
            loop.call_soon_threadsafe(queue.put_nowait, {
                "done": True, "text": decoder.text, "token_count": token_count,
                "tokens_per_second": token_count / elapsed, "stopped": stop_event.is_set(),
            })
        except Exception as error:  # a CUDA OOM here must reach the browser, not vanish
            loop.call_soon_threadsafe(queue.put_nowait, {"error": f"{type(error).__name__}: {error}"})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    threading.Thread(target=generate_on_worker_thread, daemon=True).start()

    final_text, final_stats = "", {}
    try:
        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            if "delta" in item:
                yield sse("delta", {"text": item["delta"]})
            elif "error" in item:
                yield sse("error", {"message": item["error"]})
            elif item.get("done"):
                final_text = item["text"]
                final_stats = item
                yield sse("done", {
                    "token_count": item["token_count"],
                    "tokens_per_second": round(item["tokens_per_second"], 2),
                    "stopped": item["stopped"],
                })
            # if the browser navigated away, stop burning GPU time on a reply nobody will read
            if await http_request.is_disconnected():
                stop_event.set()
    finally:
        stop_event.set()
        with active_generations_lock:
            active_generations.pop(request_id, None)

        if final_text.strip():
            with closing(open_database()) as connection:
                connection.execute(
                    """INSERT INTO messages (session_id, role, content, timestamp, model, token_count, tokens_per_second)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (session_id, "assistant", final_text, now_iso(), manager.checkpoint_name,
                     final_stats.get("token_count"), final_stats.get("tokens_per_second")),
                )
                connection.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now_iso(), session_id))
                connection.commit()


@app.post("/api/chat")
async def api_chat(request: ChatRequest, http_request: Request):
    return StreamingResponse(
        stream_reply(request.session_id, request.message,
                     request.system_prompt, request.settings, http_request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/regenerate")
async def api_regenerate(request: RegenerateRequest, http_request: Request):
    """Drops the last assistant turn and answers the same user message
    again."""
    with closing(open_database()) as connection:
        row = connection.execute(
            "SELECT id, role FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (request.session_id,),
        ).fetchone()
        if row is not None and row["role"] == "assistant":
            connection.execute("DELETE FROM messages WHERE id = ?", (row["id"],))
            connection.commit()

    return StreamingResponse(
        stream_reply(request.session_id, None,
                     request.system_prompt, request.settings, http_request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main():
    global manager

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--host", default="127.0.0.1")
    argument_parser.add_argument("--port", type=int, default=8000)
    argument_parser.add_argument("--checkpoint", default=None,
                                  help="checkpoint filename inside ../checkpoints (default: best available chat checkpoint)")
    argument_parser.add_argument("--tokenizer-dir", default=None,
                                  help="tokenizer directory (default: ../tokenizer). A checkpoint only makes "
                                       "sense with the tokenizer it was trained on -- token ids mean nothing "
                                       "across different tokenizers.")
    argument_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = argument_parser.parse_args()

    if args.tokenizer_dir:
        global TOKENIZER_DIRECTORY
        TOKENIZER_DIRECTORY = os.path.abspath(args.tokenizer_dir)

    initialize_database()
    manager = ModelManager(args.device)

    checkpoint_name = args.checkpoint or choose_default_checkpoint()
    if checkpoint_name is None:
        print(f"no checkpoints found in {CHECKPOINT_DIRECTORY} -- the UI will start, but "
              f"chatting needs a checkpoint from train.py / sft_train.py")
    else:
        manager.load(os.path.basename(checkpoint_name))

    if os.path.isdir(STATIC_DIRECTORY):
        app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")

    import uvicorn
    print(f"\n  open http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
