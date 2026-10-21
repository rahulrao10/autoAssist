"""AutoAssist -- a lightweight AutoML-style library for supervised learning on tabular data."""
from .config import Config
from .core import AutoAssist, run

__version__ = "0.1.0"
__all__ = ["run", "AutoAssist", "Config"]
