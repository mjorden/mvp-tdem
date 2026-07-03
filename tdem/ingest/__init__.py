"""
Raw instrument-log ingest: flight directory → canonical survey CSV + sidecar.

Design: docs/RAW_INGEST.md. Stages (each a pure frame-in/frame-out function):

    readers → timesync → stack → calibrate → merge → geometry → emit

`ingest_flight()` runs one flight end-to-end; `scripts/ingest_flight.py`
wraps it for multi-flight surveys.
"""

from .pipeline import ingest_flight, ingest_survey

__all__ = ["ingest_flight", "ingest_survey"]
