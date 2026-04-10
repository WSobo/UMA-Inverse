def __getattr__(name):
    if name == "UMAInverseDataModule":
        from .datamodule import UMAInverseDataModule
        return UMAInverseDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["UMAInverseDataModule"]
