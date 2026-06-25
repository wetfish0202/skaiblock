import json
import math
import os
import re
import sqlite3
import time
from collections import defaultdict
from html import escape
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


load_dotenv()

BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
ADMIN_TELEGRAM_IDS = {
    int(x.strip())
    for x in os.environ.get("ADMIN_TELEGRAM_IDS", "").split(",")
    if x.strip()
}

MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.0-flash")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")
DATABASE_PATH = Path(os.environ.get("DATABASE_PATH", "data/skaiblock.db"))
PATCHNOTES_PATH = Path(os.environ.get("PATCHNOTES_PATH", "data/patchnotes"))

SOURCE_LINK_LIMIT = int(os.environ.get("SOURCE_LINK_LIMIT", "5"))
MAX_CONTEXT_CHUNKS = int(os.environ.get("MAX_CONTEXT_CHUNKS", "8"))
USER_QUESTION_LIMIT = int(os.environ.get("USER_QUESTION_LIMIT", "50"))
USER_WINDOW_SECONDS = int(os.environ.get("USER_WINDOW_SECONDS", "600"))

TELEGRAM_LIMIT = 3900

SYSTEM_PROMPT = """You are SK(AI) BLOCK, a polished Hypixel SkyBlock patch-note expert.

Answer using only the supplied source excerpts. Be helpful, complete, and clear.
If the sources do not contain enough information, say what is known and what is not established by the available patch notes.
Prefer concrete dates, update names, changed mechanics, and cause/effect explanations.
Do not invent patch notes, dates, links, item stats, or balance changes.
Do not include a resources section; the bot adds resources separately."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS patchnote_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    published_at TEXT,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_patchnote_chunks_source
ON patchnote_chunks(source_id);
"""

rate_history = defaultdict(list)

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def require_config():
    if not BOT_TOKEN:
        raise RuntimeError("Missing BOT_TOKEN environment variable.")
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")


def connect_db() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def html_message(text: str) -> dict:
    return {
        "text": text,
        "parse_mode": ParseMode.HTML,
        "disable_web_page_preview": True,
    }


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]

    parts = []
    remaining = text
    while len(remaining) > limit:
        split_at = max(remaining.rfind("\n\n", 0, limit), remaining.rfind("\n", 0, limit))
        if split_at < limit * 0.5:
            split_at = limit
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        parts.append(remaining)
    return parts


def check_rate_limit(user_id: int) -> tuple[bool, str]:
    now = time.time()
    history = [t for t in rate_history[user_id] if now - t < USER_WINDOW_SECONDS]
    rate_history[user_id] = history

    if len(history) < USER_QUESTION_LIMIT:
        history.append(now)
        return True, ""

    resets_in = USER_WINDOW_SECONDS - (now - min(history))
    mins = int(resets_in // 60)
    secs = int(resets_in % 60)
    reset_text = f"{mins}m {secs}s" if mins else f"{secs}s"
    return False, f"You reached {USER_QUESTION_LIMIT} questions. Try again in {reset_text}."


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 250) -> list[str]:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        window = text[start:end]
        split_at = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))

        if split_at > max_chars * 0.45:
            end = start + split_at + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(0, end - overlap)

    return chunks


def source_id_from_path(path: Path) -> str:
    return path.stem.lower().replace(" ", "-")


def read_patchnote(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "source_id": data.get("source_id") or source_id_from_path(path),
            "title": data["title"],
            "url": data.get("url", ""),
            "published_at": data.get("published_at", data.get("date", "")),
            "content": normalize_text(data["content"]),
        }

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    meta = {}
    body_start = 0

    for index, line in enumerate(lines[:10]):
        if ":" not in line:
            body_start = index
            break
        key, value = line.split(":", 1)
        meta[key.strip().lower()] = value.strip()
        body_start = index + 1

    return {
        "source_id": meta.get("source_id") or source_id_from_path(path),
        "title": meta.get("title") or path.stem,
        "url": meta.get("url", ""),
        "published_at": meta.get("date", meta.get("published_at", "")),
        "content": normalize_text("\n".join(lines[body_start:])),
    }


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = get_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return [e.values for e in response.embeddings]


async def ingest_patchnotes() -> int:
    PATCHNOTES_PATH.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path for path in PATCHNOTES_PATH.glob("*")
        if path.suffix.lower() in {".txt", ".md", ".json"}
    )

    rows = []

    for path in files:
        source = read_patchnote(path)
        chunks = chunk_text(source["content"])
        embeddings = await embed_texts(chunks)

        for index, (content, embedding) in enumerate(zip(chunks, embeddings)):
            rows.append({
                "source_id": source["source_id"],
                "title": source["title"],
                "url": source["url"],
                "published_at": source["published_at"],
                "chunk_index": index,
                "content": content,
                "embedding": embedding,
            })

    conn = connect_db()
    try:
        conn.execute("DELETE FROM patchnote_chunks")
        for row in rows:
            conn.execute(
                """
                INSERT INTO patchnote_chunks
                (source_id, title, url, published_at, chunk_index, content, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["source_id"],
                    row["title"],
                    row["url"],
                    row["published_at"],
                    row["chunk_index"],
                    row["content"],
                    json.dumps(row["embedding"]),
                ),
            )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def chunk_count() -> int:
    conn = connect_db()
    try:
        row = conn.execute("SELECT COUNT(*) AS count FROM patchnote_chunks").fetchone()
        return int(row["count"])
    finally:
        conn.close()


def load_chunks() -> list[dict]:
    conn = connect_db()
    try:
        rows = conn.execute(
            """
            SELECT id, source_id, title, url, published_at, chunk_index, content, embedding
            FROM patchnote_chunks
            """
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            **dict(row),
            "embedding": json.loads(row["embedding"]),
        }
        for row in rows
    ]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if not mag_a or not mag_b:
        return 0.0
    return dot / (mag_a * mag_b)


async def retrieve_chunks(question: str) -> list[dict]:
    client = get_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_embedding = response.embeddings[0].values

    scored = []
    for chunk in load_chunks():
        scored.append({
            **chunk,
            "score": cosine_similarity(query_embedding, chunk["embedding"]),
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:MAX_CONTEXT_CHUNKS]


def build_context(chunks: list[dict]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, 1):
        date = chunk.get("published_at") or "unknown date"
        title = chunk.get("title") or "Untitled"
        parts.append(f"[Source {index}: {title} | {date}]\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


def build_resources(chunks: list[dict]) -> str:
    seen = set()
    links = []

    for chunk in chunks:
        key = (chunk.get("title"), chunk.get("url"))
        if key in seen or not chunk.get("url"):
            continue
        seen.add(key)
        title = escape(chunk["title"])
        url = escape(chunk["url"], quote=True)
        links.append(f'<a href="{url}">{title}</a>')
        if len(links) >= SOURCE_LINK_LIMIT:
            break

    if not links:
        return ""

    return "<b>Resources:</b> " + " | ".join(links)


async def answer_question(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return (
            "I do not have any patch-note sources loaded yet. Add patch notes to "
            f"<code>{PATCHNOTES_PATH}</code>, then run <code>/reload</code>."
        )

    client = get_client()
    context = build_context(chunks)
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question:\n{question}\n\n"
        f"Patch-note excerpts:\n{context}"
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )

    answer = escape(response.text.strip())
    resources = build_resources(chunks)
    return f"{answer}\n\n{resources}" if resources else answer


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_TELEGRAM_IDS


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "<b>SK(AI) BLOCK</b>\n\n"
        "Ask me Hypixel SkyBlock questions and I will answer using loaded patch notes, "
        "then add clickable resources at the bottom.\n\n"
        "Try: <code>What changed in the Garden update?</code>"
    )
    await update.message.reply_text(**html_message(text))


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "<b>Commands</b>\n\n"
        "/start - intro\n"
        "/help - command list\n"
        "/status - knowledge base status\n"
        "/reload - admin only, re-ingest patch notes\n\n"
        "Or just send a SkyBlock question."
    )
    await update.message.reply_text(**html_message(text))


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "<b>SK(AI) BLOCK Status</b>\n\n"
        f"Loaded chunks: <code>{chunk_count()}</code>\n"
        f"Patch notes folder: <code>{PATCHNOTES_PATH}</code>\n"
        f"Model: <code>{MODEL_NAME}</code>\n"
        f"Question limit: <code>{USER_QUESTION_LIMIT} per {USER_WINDOW_SECONDS}s</code>"
    )
    await update.message.reply_text(**html_message(text))


async def reload_sources(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("Admin only.")
        return

    msg = await update.message.reply_text("Reloading patch notes...")
    try:
        count = await ingest_patchnotes()
    except Exception as exc:
        await msg.edit_text(f"Reload failed: {exc}")
        return

    await msg.edit_text(f"Reload complete. Loaded {count} chunks.")


async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    question = update.message.text.strip()

    if len(question) > 1500:
        await update.message.reply_text("That question is too long. Keep it under 1500 characters.")
        return

    allowed, reason = check_rate_limit(user_id)
    if not allowed:
        await update.message.reply_text(reason)
        return

    thinking = await update.message.reply_text("Thinking...")

    try:
        chunks = await retrieve_chunks(question)
        answer = await answer_question(question, chunks)
    except Exception as exc:
        await thinking.edit_text(f"I hit an error while answering: {exc}")
        return

    parts = split_message(answer)
    await thinking.edit_text(**html_message(parts[0]))

    for part in parts[1:]:
        await update.message.reply_text(**html_message(part))


async def post_init(app: Application):
    PATCHNOTES_PATH.mkdir(parents=True, exist_ok=True)

    if chunk_count() > 0:
        print(f"[startup] loaded existing database with {chunk_count()} chunks")
        return

    try:
        count = await ingest_patchnotes()
        print(f"[startup] ingested {count} patch-note chunks")
    except Exception as exc:
        print(f"[startup] ingestion skipped/failed: {exc}")


def main():
    require_config()

    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("reload", reload_sources))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    print("SK(AI) BLOCK running...")
    app.run_polling()


if __name__ == "__main__":
    main()
