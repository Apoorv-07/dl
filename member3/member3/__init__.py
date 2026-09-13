"""Member 3 - adaptive memory, FAISS retrieval, confidence-aware thresholding,
multi-stage alert verification, graph attribution and evaluation.

Modules
-------
``contract``   Member-1/2 handoff interface, record I/O, contamination guards, masking
``memory``     :class:`AdaptiveMemoryBank` (+ FAISS HNSW front-end)
``inference``  anomaly score, threshold calibration, alert verification, latency
``explain``    GATv2 attribution orchestration and Fidelity+ validation
``evaluate``   metrics, bootstrap CIs, ablations, paper tables and figures
``api``        minimal FastAPI service + analyst view
"""

__version__ = "0.1.0"

from .inference import AlertTracker, Member3Scorer, ThresholdModel, calibrate_threshold  # noqa: F401
from .memory import AdaptiveMemoryBank  # noqa: F401
