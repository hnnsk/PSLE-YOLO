"""
Compatibility shim for historical checkpoints.

Older local training runs serialized custom modules under
`ultralytics.nn.modules.mafyolo`. The current project keeps those symbols in
`ultralytics.nn.modules.C3STR`, so this file re-exports them to keep old
checkpoints loadable without modifying the weights.
"""

from .C3STR import *  # noqa: F401,F403
