"""Optional read-only integration contracts against the live Qdrant collection.

These tests never write, upsert or delete. They are skipped unless
``RUN_QDRANT_INTEGRATION=1`` is set, so the normal unit suite stays offline.
"""
