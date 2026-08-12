"""Optional experiment tracking with a no-op default.

SwanLab is deliberately imported only when enabled, so training and tests do
not acquire a hard network or authentication dependency.  In DDP mode rank 0
logs metrics that have already been reduced across ranks.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SwanLabConfig:
    enabled: bool = False
    project: str = "VALLR-Pin"
    workspace: str = ""
    experiment_name: str = ""
    description: str = ""
    mode: str = "disabled"  # online | offline | local | disabled
    logdir: str = "swanlog"
    tags: List[str] = field(default_factory=list)
    run_id: str = ""
    resume: bool | str = False  # False | True | allow | must


def _serializable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: _serializable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    return value


class SwanLabTracker:
    """Small rank-safe adapter around SwanLab's init/log/finish API."""

    def __init__(self, cfg: SwanLabConfig, run_config: Any, is_main: bool = True):
        self.cfg, self.run, self.module = cfg, None, None
        if not cfg.enabled or not is_main:
            return
        if cfg.mode not in {"online", "offline", "local", "disabled"}:
            raise ValueError(f"unsupported SwanLab mode: {cfg.mode}")
        if cfg.resume not in {False, True, "allow", "must"}:
            raise ValueError("swanlab.resume must be false, true, allow, or must")
        if cfg.resume and not cfg.run_id:
            raise ValueError("swanlab.run_id is required when resume is enabled")
        try:
            import swanlab
        except ImportError as error:
            raise ImportError(
                "SwanLab tracking is enabled but swanlab is not installed; "
                "run `pip install swanlab` or set swanlab.enabled=false"
            ) from error
        kwargs: Dict[str, Any] = {
            "project": cfg.project,
            "experiment_name": cfg.experiment_name or None,
            "description": cfg.description or None,
            "mode": cfg.mode,
            "logdir": cfg.logdir,
            "tags": cfg.tags or None,
            "config": _serializable(run_config),
        }
        if cfg.workspace:
            kwargs["workspace"] = cfg.workspace
        if cfg.run_id:
            kwargs["id"] = cfg.run_id
            # Passing resume=False together with an id means "never" in
            # SwanLab and is rejected.  Omit it for a fresh custom-id run.
            if cfg.resume:
                kwargs["resume"] = cfg.resume
        self.module = swanlab
        self.run = swanlab.init(**{k: v for k, v in kwargs.items() if v is not None})

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log(self, values: Dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            return
        metrics = {k: v for k, v in values.items() if v is not None}
        if metrics:
            self.module.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.module.finish()
            self.run = None
