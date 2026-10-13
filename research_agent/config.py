"""Environment-backed configuration and credential validation."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    serper_api_key: str
    openai_model: str = "gpt-4o"
    log_level: str = "INFO"
    output_dir: Path = Path("output")

    @classmethod
    def from_env(cls, require_keys: bool = True) -> "Settings":
        load_dotenv()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        serper_key = os.getenv("SERPER_API_KEY", "").strip()
        if require_keys:
            missing = []
            if not openai_key:
                missing.append("OPENAI_API_KEY")
            if not serper_key:
                missing.append("SERPER_API_KEY")
            if missing:
                raise ValueError(
                    "Missing required environment variable(s): " + ", ".join(missing)
                    + ". Copy .env.example to .env and add your API keys."
                )
        output_value = os.getenv("OUTPUT_DIR", "output").strip() or "output"
        return cls(
            openai_api_key=openai_key,
            serper_api_key=serper_key,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o",
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
            output_dir=Path(output_value),
        )
