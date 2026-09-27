from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

from backend.logger import get_logger
from database.db_manager import mark_alert_sent, queue_alert

logger = get_logger("alerts")
BASE_DIR = Path(__file__).resolve().parents[1]
OUTBOX_DIR = BASE_DIR / "logs" / "alert_outbox"
OUTBOX_DIR.mkdir(parents=True, exist_ok=True)


def _smtp_settings() -> Optional[dict]:
    host = os.getenv("OVERSPEED_SMTP_HOST")
    port = os.getenv("OVERSPEED_SMTP_PORT")
    user = os.getenv("OVERSPEED_SMTP_USER")
    password = os.getenv("OVERSPEED_SMTP_PASSWORD")
    sender = os.getenv("OVERSPEED_SMTP_SENDER") or user
    if not (host and port and sender):
        return None
    return {"host": host, "port": int(port), "user": user, "password": password, "sender": sender}


def send_email_alert(recipient: str, subject: str, body: str) -> int:
    alert_id = queue_alert("email", recipient, subject, body)
    cfg = _smtp_settings()
    if not cfg:
        file_path = OUTBOX_DIR / f"email_{alert_id}.txt"
        file_path.write_text(f"TO: {recipient}\nSUBJECT: {subject}\n\n{body}", encoding="utf-8")
        mark_alert_sent(alert_id, "sent", None)
        logger.info("Email alert written locally to %s", file_path)
        return alert_id

    try:
        msg = EmailMessage()
        msg["From"] = cfg["sender"]
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=15) as server:
            server.starttls()
            if cfg["user"] and cfg["password"]:
                server.login(cfg["user"], cfg["password"])
            server.send_message(msg)
        mark_alert_sent(alert_id, "sent")
        logger.info("Email alert sent to %s", recipient)
    except Exception as exc:
        mark_alert_sent(alert_id, "failed", str(exc))
        logger.exception("Email alert failed")
    return alert_id


def send_sms_alert(recipient: str, body: str) -> int:
    """Zero-cost fallback: writes SMS alerts to local outbox file unless a provider is added later."""
    alert_id = queue_alert("sms", recipient, "SMS Alert", body)
    file_path = OUTBOX_DIR / f"sms_{alert_id}.txt"
    file_path.write_text(f"TO: {recipient}\n\n{body}", encoding="utf-8")
    mark_alert_sent(alert_id, "sent")
    logger.info("SMS alert written locally to %s", file_path)
    return alert_id
