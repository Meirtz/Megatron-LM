"""Native GLM-5 lite implementation."""

__all__ = ["Glm5Model"]


def __getattr__(name: str):
    if name == "Glm5Model":
        from megatron.lite.model.glm5.lite.model import Glm5Model

        return Glm5Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
