"""Streaming engine for batch-wise inference and adaptive learning."""

__all__ = ["StreamingEngine", "StreamArtifacts"]


def __getattr__(name):
    if name in __all__:
        from sa_zd_nids.streaming.engine import StreamingEngine, StreamArtifacts  # noqa: F401
        return {"StreamingEngine": StreamingEngine, "StreamArtifacts": StreamArtifacts}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
