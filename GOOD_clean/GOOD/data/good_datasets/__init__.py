"""Register only the node-level GOOD datasets needed in this workspace.

The full upstream package imports every dataset eagerly, which pulls in optional
dependencies for graph tasks that we are not using here. Restrict the imports
to the node-level datasets used by our local GOOD experiments.
"""

from .good_arxiv import GOODArxiv
from .good_cbas import GOODCBAS
from .good_cora import GOODCora
from .good_twitch import GOODTwitch
from .good_webkb import GOODWebKB

__all__ = [
    "GOODArxiv",
    "GOODCBAS",
    "GOODCora",
    "GOODTwitch",
    "GOODWebKB",
]
