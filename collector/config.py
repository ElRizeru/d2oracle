"""Application settings for the d2oracle match collector."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CollectorSettings(BaseSettings):
    """Configuration loaded from environment variables and local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    stratz_token: SecretStr
    hf_token: SecretStr
    hf_repo_id: str = "rizeru/d2oracle-data"

    target_game_version: str = "auto"

    batch_size: int = Field(default=5000, ge=1)
    sqlite_path: Path = Path("data/d2oracle.sqlite3")
    parquet_dir: Path = Path("data/shards")
    constants_cache_dir: Path = Path("data/constants")

    patch_check_interval_seconds: int = Field(default=3600, ge=60)
    pro_discovery_interval_seconds: int = Field(default=1800, ge=60)
    public_discovery_interval_seconds: int = Field(default=1800, ge=60)

    # Safe STRATZ rate ceilings (hard API limits: 8/s, 150/m, 1500/h, 15000/d)
    stratz_rps_limit: int = Field(default=6, ge=1, le=8)
    stratz_rpm_limit: int = Field(default=130, ge=1, le=150)
    stratz_rph_limit: int = Field(default=1350, ge=1, le=1500)
    stratz_rpd_limit: int = Field(default=14000, ge=1, le=15000)

    # Safe OpenDota rate ceilings (unauthenticated limits: 60/m, 3000/d)
    opendota_rpm_limit: int = Field(default=50, ge=1, le=60)
    opendota_rpd_limit: int = Field(default=2800, ge=1, le=3000)

    # Discovery lag windows so STRATZ has time to parse matches
    public_match_min_delay_seconds: int = Field(default=3600, ge=0)
    public_match_max_delay_seconds: int = Field(default=10800, ge=60)
    pro_match_min_delay_seconds: int = Field(default=1800, ge=0)

    player_bfs_batch_size: int = Field(default=10, ge=1, le=10)
    player_bfs_cooldown_seconds: int = Field(default=14400, ge=0)
    player_bfs_max_age_seconds: int = Field(
        default=0,
        ge=0,
        description="Max age of matches to discover via BFS in seconds; 0 means entire active patch from as_of_datetime.",
    )
    match_retry_cooldown_seconds: int = Field(default=1800, ge=0, le=86400)

    log_level: str = "INFO"

    @field_validator("target_game_version")
    @classmethod
    def validate_target_game_version(cls, v: str) -> str:
        clean = v.strip()
        if clean.lower() == "auto":
            return "auto"
        try:
            gv = int(clean)
        except ValueError as exc:
            raise ValueError(
                "TARGET_GAME_VERSION must be 'auto' or a positive integer gameVersionId"
            ) from exc
        if gv <= 0:
            raise ValueError("TARGET_GAME_VERSION integer ID must be positive")
        return str(gv)
