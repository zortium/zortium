from __future__ import annotations

import os
from enum import Enum
from pathlib import Path


class DeploymentMode(str, Enum):
    CLOUD = "cloud"
    ONPREM = "onprem"


class Settings:

    @property
    def deployment_mode(self) -> DeploymentMode:
        raw = os.getenv("DEPLOYMENT_MODE", "cloud").strip().lower()
        try:
            return DeploymentMode(raw)
        except ValueError:
            return DeploymentMode.CLOUD

    @property
    def license_key(self) -> str:
        return os.getenv("LICENSE_KEY", "").strip()

    @property
    def api_url(self) -> str:
        return os.getenv("ZORTIUM_API_URL", "").strip().rstrip("/")

    @property
    def target_api_key(self) -> str:
        return os.getenv("ZORTIUM_API_KEY", "").strip()

    @property
    def target_model(self) -> str:
        return os.getenv("ZORTIUM_MODEL", "").strip()

    @property
    def target_base_url(self) -> str:
        return os.getenv("ZORTIUM_BASE_URL", "").strip().rstrip("/")

    @property
    def judge_api_key(self) -> str:
        return os.getenv("JUDGE_API_KEY", "").strip()

    @property
    def judge_model(self) -> str:
        return os.getenv("JUDGE_MODEL", "").strip() or "gpt-4o-mini"

    @property
    def judge_base_url(self) -> str:
        return (os.getenv("JUDGE_BASE_URL", "").strip() or "https://api.openai.com/v1").rstrip("/")

    @property
    def attacker_api_key(self) -> str:
        return os.getenv("ATTACKER_API_KEY", "").strip() or self.judge_api_key

    @property
    def attacker_model(self) -> str:
        return os.getenv("ATTACKER_MODEL", "").strip() or self.judge_model

    @property
    def attacker_base_url(self) -> str:
        return os.getenv("ATTACKER_BASE_URL", "").strip().rstrip("/") or self.judge_base_url

    @property
    def typesafe_api_key(self) -> str:
        return os.getenv("TYPESAFE_API_KEY", "").strip()

    @property
    def jev_model(self) -> str:
        return os.getenv("JEV_MODEL", "").strip() or "jev-1.13.0"

    @property
    def scan_timeout_minutes(self) -> int:
        try:
            return max(1, int(os.getenv("ZORTIUM_SCAN_TIMEOUT_MINUTES", "60")))
        except ValueError:
            return 60

    @property
    def max_concurrent_scans(self) -> int:
        try:
            return max(1, int(os.getenv("ZORTIUM_MAX_CONCURRENT_SCANS", "2")))
        except ValueError:
            return 2

    @property
    def token_expire_minutes(self) -> int:
        try:
            return max(1, int(os.getenv("ZORTIUM_TOKEN_EXPIRE_MINUTES", "1440")))
        except ValueError:
            return 1440

    @property
    def free_weekly_quota(self) -> int:
        try:
            return max(0, int(os.getenv("ZORTIUM_FREE_WEEKLY_QUOTA", "2")))
        except ValueError:
            return 2

    @property
    def demo_daily_quota(self) -> int:
        try:
            return max(0, int(os.getenv("ZORTIUM_DEMO_DAILY_QUOTA", "10")))
        except ValueError:
            return 10

    @property
    def google_client_id(self) -> str:
        return os.getenv("GOOGLE_CLIENT_ID", "").strip()

    @property
    def google_client_secret(self) -> str:
        return os.getenv("GOOGLE_CLIENT_SECRET", "").strip()

    @property
    def oauth_enabled(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)

    @property
    def google_signup_enabled(self) -> bool:
        return os.getenv("ZORTIUM_GOOGLE_SIGNUP", "true").strip().lower() != "false"

    @property
    def public_url(self) -> str:
        # Absolute origin (e.g. https://zortium.dev) used to build the OAuth
        # redirect URI behind a TLS-terminating proxy, where request.url would
        # otherwise report http and break Google's exact-match check.
        return os.getenv("ZORTIUM_PUBLIC_URL", "").strip().rstrip("/")

    @property
    def config_dir(self) -> Path:
        """Per-user CLI settings folder: %APPDATA%\\zortium on Windows, else $XDG_CONFIG_HOME/zortium (~/.config/zortium)."""
        if os.name == "nt" and os.getenv("APPDATA", "").strip():
            return Path(os.getenv("APPDATA", "").strip()) / "zortium"

        base = os.getenv("XDG_CONFIG_HOME", "").strip()
        return (Path(base) if base else Path.home() / ".config") / "zortium"

    @property
    def session_secret(self) -> str:
        return os.getenv("ZORTIUM_SESSION_SECRET", "").strip() or os.getenv("JWT_SECRET_KEY", "").strip()


settings = Settings()
