import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class _RankContext:
    rank: int
    world_size: int


class _RankFilter(logging.Filter):
    def __init__(self, ctx: _RankContext):
        super().__init__()
        self._ctx = ctx

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        # Inject rank fields so formatters can use %(rank)s / %(world_size)s.
        record.rank = self._ctx.rank
        record.world_size = self._ctx.world_size
        return True


def setup_rank_logging(
    output_path: str,
    rank: int,
    world_size: int,
    *,
    level: int = logging.INFO,
    log_subdir: str = "logs",
    stream_rank0_only: bool = True,
    stream: Optional[object] = None,
) -> str:
    """
    Configure process-safe logging for distributed runs:
    - always write to per-rank file: {output_path}/{log_subdir}/rank_{rank}.log
    - only stream to console on rank 0 (default)

    Returns the absolute log file path.
    """
    if stream is None:
        stream = sys.stdout

    os.makedirs(output_path, exist_ok=True)
    log_dir = os.path.join(output_path, log_subdir)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.abspath(os.path.join(log_dir, f"rank_{rank}.log"))
    ctx = _RankContext(rank=rank, world_size=world_size)
    rank_filter = _RankFilter(ctx)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | R%(rank)d/%(world_size)d | %(name)s | %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Drop console handlers on non-rank0 to keep tmux readable.
    allow_stream = (rank == 0) if stream_rank0_only else True
    if not allow_stream:
        for h in list(root.handlers):
            if isinstance(h, logging.StreamHandler):
                root.removeHandler(h)

    # Add (or reuse) per-rank file handler.
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, "baseFilename", "")) == log_file:
            return log_file

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    file_handler.addFilter(rank_filter)
    root.addHandler(file_handler)

    if allow_stream:
        stream_handler = logging.StreamHandler(stream=stream)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(fmt)
        stream_handler.addFilter(rank_filter)
        root.addHandler(stream_handler)

    # Common noisy libs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    return log_file


