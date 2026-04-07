from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from email import message_from_bytes
from pathlib import Path

from aiosmtpd.controller import Controller


class SinkHandler:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.out_dir / "messages.jsonl"

    async def handle_DATA(self, server, session, envelope) -> str:
        message_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        raw_path = self.out_dir / f"{message_id}.eml"
        raw_path.write_bytes(envelope.content)

        parsed = message_from_bytes(envelope.content)
        record = {
            "message_id": message_id,
            "mail_from": envelope.mail_from,
            "rcpt_tos": envelope.rcpt_tos,
            "peer": session.peer,
            "subject": parsed.get("Subject", ""),
            "date": parsed.get("Date", ""),
            "saved_path": str(raw_path),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[smtp-sink] captured {message_id} -> {record['subject']}")
        return "250 Message accepted for delivery"


def main() -> None:
    parser = argparse.ArgumentParser(description="Local SMTP sink server for TicketFlow MCP tests.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1025)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "generated" / "smtp_sink",
    )
    args = parser.parse_args()

    controller = Controller(SinkHandler(args.out_dir), hostname=args.host, port=args.port)
    controller.start()
    print(f"[smtp-sink] listening on smtp://{args.host}:{args.port}")
    print(f"[smtp-sink] output dir: {args.out_dir}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
