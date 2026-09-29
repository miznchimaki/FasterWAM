from .fs import ensure_dir

__all__ = ["ensure_dir", "save_mp4"]


def __getattr__(name):
    # Configuration-only entrypoints do not need the video I/O dependencies.
    if name == "save_mp4":
        from .video_io import save_mp4

        return save_mp4
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
