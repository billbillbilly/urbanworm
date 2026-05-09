"""Top-level package for urbanworm."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

__author__ = "Xiaohao Yang"
__email__ = "xiaohaoy111@gmail.com"

try:
    __version__ = _pkg_version("urban-worm")
except PackageNotFoundError:  # package is not installed (e.g. running from source)
    __version__ = "0.0.0+unknown"

# from .inference.transformers import InferenceTrans
from .dataset import GeoTaggedData, getPhoto, getSound, getSV
from .inference.llama import InferenceLlamacpp, InferenceOllama


def __getattr__(name: str):
    """Lazily expose InferenceUnsloth so importing urbanworm does not
    trigger the heavy torch/unsloth stack unless the class is requested."""
    if name == "InferenceUnsloth":
        from .inference.unsloth import InferenceUnsloth as _IU
        return _IU
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "InferenceOllama",
    "InferenceLlamacpp",
    "InferenceUnsloth",
    "GeoTaggedData",
    "getSV",
    "getPhoto",
    "getSound",
]
