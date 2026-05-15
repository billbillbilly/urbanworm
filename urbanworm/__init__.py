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
    """Lazily expose heavy optional backends so importing urbanworm does not
    pull in torch/unsloth or provider SDKs unless the class is requested."""
    if name == "InferenceUnsloth":
        from .inference.unsloth import InferenceUnsloth as _IU
        return _IU
    if name == "InferenceAPI":
        from .inference.api import InferenceAPI as _IA
        return _IA
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "InferenceOllama",
    "InferenceLlamacpp",
    "InferenceUnsloth",
    "InferenceAPI",
    "GeoTaggedData",
    "getSV",
    "getPhoto",
    "getSound",
]
