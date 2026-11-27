"""Dataset loading and validation.

The YAML box dataset declaration lives in :mod:`polyx.core.data.yolo`: it owns
``data.yaml`` parsing, the dataset root, each split's location and membership,
and the ordered class names, and passes the names to the YOLO-label codec
(``polyx.core.labels``) as ``class_names``. The package holds no label I/O of
its own: the codec is the single owner of the ``.txt`` format and its folder
conventions.

The canonical COCO mask dataset keeps its own reader
(``polyx.pipeline.coco``/``polyx.pipeline.dataset``); its declaration is not a
YAML file and is deliberately not forced behind this interface.
"""

from polyx.core.data.yolo import (
    SPLIT_NAMES,
    DatasetDeclaration,
    SplitDeclaration,
    SplitKind,
    declares_real_domain,
    write_declaration,
)

__all__ = [
    "SPLIT_NAMES",
    "DatasetDeclaration",
    "SplitDeclaration",
    "SplitKind",
    "declares_real_domain",
    "write_declaration",
]
