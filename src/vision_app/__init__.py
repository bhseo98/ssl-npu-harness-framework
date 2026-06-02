"""Vision application on the framework core — ResNet18 ImageNet classifier.

Importing this package triggers stage registration (`loader`, `preprocess`,
`classifier`, `postprocess`) so `build(...)` and `Pipeline(...)` work.
"""
from . import stages  # noqa: F401 — side-effect import registers stages

__version__ = "0.1.0"
