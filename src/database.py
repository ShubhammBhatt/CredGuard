import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path("data/creditguard.db")


def get_connection() -> sqlite3.Connection:
    """Create a connection to the SQLite database file."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the predictions log table if it does not already exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL,
                amount REAL NOT NULL,
                hour_of_day INTEGER NOT NULL,
                merchant_category TEXT NOT NULL,
                is_foreign INTEGER NOT NULL,
                is_online INTEGER NOT NULL,
                distance_from_home_km REAL NOT NULL,
                tx_count_last_24h INTEGER NOT NULL,
                card_age_months INTEGER NOT NULL,
                fraud_probability REAL NOT NULL,
                risk_decision TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def log_prediction(record: dict[str, Any]) -> int:
    """Insert one scored transaction into the predictions table and return its row ID."""
    created_at = record.get("created_at") or datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (
                transaction_id,
                amount,
                hour_of_day,
                merchant_category,
                is_foreign,
                is_online,
                distance_from_home_km,
                tx_count_last_24h,
                card_age_months,
                fraud_probability,
                risk_decision,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["transaction_id"],
                float(record["amount"]),
                int(record["hour_of_day"]),
                str(record["merchant_category"]),
                int(record["is_foreign"]),
                int(record["is_online"]),
                float(record["distance_from_home_km"]),
                int(record["tx_count_last_24h"]),
                int(record["card_age_months"]),
                float(record["fraud_probability"]),
                str(record["risk_decision"]),
                created_at,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def fetch_recent_predictions(limit: int = 25) -> list[dict[str, Any]]:
    """Fetch the most recent logged predictions, newest first."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]