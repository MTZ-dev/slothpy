from __future__ import annotations

from slothpy.core.slt_progress import SltProgressTracker
from slothpy.io.shared_memory import (
    release_all_parent_owned_shared_blocks,
    unregister_parent_owned_shared_block,
)


def test_progress_tracker_registers_and_releases_parent_memory() -> None:
    tracker = SltProgressTracker.create(total=4)
    name = tracker.spec.name

    try:
        unregister_parent_owned_shared_block(name)
        tracker.release()
    finally:
        release_all_parent_owned_shared_blocks()
