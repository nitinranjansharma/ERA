"""Helix — Training Data Execution System for ERA V5 Session 6.

A sealed-object, event-sourced data path:

    documents → shards → mixture → packing → OPUS → batches
      → consumption ledger → learning ledger
      → checkpoint / crash / resume / replay / fork / audit
"""

__version__ = "1.0.0"
