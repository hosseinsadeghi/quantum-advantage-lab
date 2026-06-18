Repo-backed run cache.

- `.cache/runs/*.jsonl` stores replayable quantum run results that may be committed.
- `QAL_CACHE_DIR` can still override this location for ad hoc runs or tests.
- `.cache/qpu_usage.jsonl` remains a local-only usage log and is ignored by git.
