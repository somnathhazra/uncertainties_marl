REGISTRY = {}

from .basic_controller import BasicMAC
from .n_iql_dist_controller import NIQLDistMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_iql_dist_mac"] = NIQLDistMAC
