import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

from unittest.mock import MagicMock

# 1. Imports must succeed
from sglang.srt.managers.scheduler_components.finetune_coordinator import FinetuneCoordinator
from sglang.srt.managers.scheduler_components.finetune_scheduler_mixin import FinetuneSchedulerMixin
from sglang.srt.configs.finetune import FinetuneConfig

# 2. Construct coordinator with real FinetuneConfig
cfg = FinetuneConfig(enable_finetuning=True)
coord = FinetuneCoordinator(cfg, backward_channel=None)

# 3. reserve(100) -> True
assert coord.reserve(100) is True, "reserve(100) should return True"

# 4. note_injection with MagicMock — no exception
coord.note_injection(MagicMock())

# 5. pause/resume no-ops (channel is None)
coord.gpu_pause_backward()
coord.gpu_resume_backward()

# 6. on_backward_done callable
coord.on_backward_done({})

# 7. Mixin surface present
assert hasattr(FinetuneSchedulerMixin, "_select_batch")
assert hasattr(FinetuneSchedulerMixin, "process_input_requests")
assert hasattr(FinetuneSchedulerMixin, "event_loop_normal")

print("phase7 ok")
