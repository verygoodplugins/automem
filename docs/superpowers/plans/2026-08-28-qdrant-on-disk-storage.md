# Qdrant On-Disk Storage Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task.

**Goal:** Make every newly created AutoMem Qdrant collection disk-backed for vectors, HNSW, and payloads, and provide an explicit, documented Qdrant-only restore path for migrating existing collections.

**Architecture:** Collection creation is the only runtime configuration mutation: it supplies Qdrant's \`on_disk\` options when the collection does not already exist. Existing collections are untouched. The backup restore tool accepts an explicit HNSW disk-storage environment variable so operators can rebuild a selected Qdrant collection from a portable backup; docs make that destructive operation, verification, and manual Railway volume resizing explicit.

**Tech Stack:** Python 3.12, qdrant-client, Flask, pytest, Bash, Markdown.

## Global Constraints

- Preserve the configuration of any existing Qdrant collection; never update it at API startup.
- Keep production migration opt-in and operator-driven. Do not automate a Railway volume resize.
- Keep the restore script's existing explicit \`--force\` confirmation model and its \`--qdrant-only\` scope.
- Treat a Qdrant backup as sensitive corpus data; the runbook must use environment variables for credentials and avoid embedding secrets.

---

### Task 1: Set safe on-disk defaults for newly created collections

**Files:**
- Modify: \`automem/stores/runtime_clients.py\`
- Modify: \`automem/service_runtime_bindings.py\`
- Modify: \`app.py\`
- Modify: \`tests/test_vector_size_safety.py\`

**Step 1: Add the failing collection-creation regression test.**

In \`TestEnsureQdrantCollectionPayloadIndexes\`, add a test for a missing collection. Call \`ensure_qdrant_collection\` with the Qdrant model shims, assert one \`create_collection\` call, and verify that:

- \`vectors_config\` uses the resolved embedding dimension and \`Distance.COSINE\`;
- \`vectors_config.on_disk is True\`;
- \`hnsw_config.on_disk is True\`; and
- \`on_disk_payload is True\`.

Also assert the existing payload indexes are still created after the collection is created. Keep the pre-existing collection test proving that initialization does not recreate or reconfigure an existing collection.

**Step 2: Run the focused test to confirm the new assertion fails.**

Run:

\`\`\`bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/jgarturo/Projects/OpenAI/automem/.venv/bin/pytest -q tests/test_vector_size_safety.py -k on_disk
\`\`\`

Expected: failure because the current collection factory does not provide all three on-disk settings.

**Step 3: Pass \`HnswConfigDiff\` through the composition root.**

Import \`HnswConfigDiff\` together with the other Qdrant models in \`app.py\`, including both optional-import fallback branches. Add an \`hnsw_config_diff_cls\` parameter to \`create_service_runtime\`, pass it to \`ensure_qdrant_collection\`, and wire \`HnswConfigDiff\` from the \`app.py\` service runtime construction. Preserve the application's optional-dependency startup behavior.

**Step 4: Configure only newly created collections.**

Extend \`ensure_qdrant_collection\` to accept the HNSW configuration class. In its existing missing-collection branch, create the collection with:

\`\`\`python
vectors_config=VectorParams(size=effective_dim, distance=Distance.COSINE, on_disk=True)
hnsw_config=HnswConfigDiff(on_disk=True)
on_disk_payload=True
\`\`\`

Do not alter the early return for an existing collection or the vector-size mismatch handling.

**Step 5: Run focused tests.**

Run:

\`\`\`bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/jgarturo/Projects/OpenAI/automem/.venv/bin/pytest -q tests/test_vector_size_safety.py
\`\`\`

Expected: all vector-size safety tests pass, including the new missing-collection regression case.

**Step 6: Commit the runtime change.**

\`\`\`bash
git add app.py automem/stores/runtime_clients.py automem/service_runtime_bindings.py tests/test_vector_size_safety.py
git commit -m "perf(qdrant): store new collection data on disk"
\`\`\`

---

### Task 2: Support explicit disk-backed HNSW during Qdrant restores

**Files:**
- Modify: \`scripts/restore_from_backup.py\`
- Modify: \`scripts/lab/clone_production.sh\`
- Modify: \`tests/test_backup_endpoint.py\`

**Step 1: Extend the restore configuration test first.**

In \`test_restore_qdrant_uses_optional_collection_tuning\`, monkeypatch a new \`QDRANT_RESTORE_HNSW_ON_DISK\` setting to \`True\`. Assert the \`HnswConfigDiff\` used for \`create_collection\` contains both the existing \`m=0\` option and \`on_disk is True\`. Add a focused lab-clone-script assertion for a \`QDRANT_RESTORE_HNSW_ON_DISK\` default of \`true\`.

**Step 2: Run the targeted test to confirm it fails.**

Run:

\`\`\`bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/jgarturo/Projects/OpenAI/automem/.venv/bin/pytest -q tests/test_backup_endpoint.py -k "optional_collection_tuning or lab_clone_uses_paced_qdrant_restore_defaults"
\`\`\`

Expected: failure because the restore module and local-clone wrapper do not yet expose the HNSW disk setting.

**Step 3: Add the optional restore environment setting.**

Define \`QDRANT_RESTORE_HNSW_ON_DISK = _optional_bool_env("QDRANT_RESTORE_HNSW_ON_DISK")\` next to the other restore collection tuning values. Refactor \`_qdrant_collection_kwargs\` to build an HNSW keyword dictionary, adding \`m\` when configured and \`on_disk\` when configured, and instantiate \`HnswConfigDiff(**hnsw_kwargs)\` only when at least one setting is supplied. This must preserve \`m=0\` as an intentional value.

**Step 4: Set the local production-clone default.**

Pass \`QDRANT_RESTORE_HNSW_ON_DISK="\${QDRANT_RESTORE_HNSW_ON_DISK:-true}"\` in \`scripts/lab/clone_production.sh\`, adjacent to the existing HNSW/vector/payload restore settings. This makes cloned production data match the new disk-backed default without affecting a running production collection.

**Step 5: Run the focused restore tests.**

Run:

\`\`\`bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/jgarturo/Projects/OpenAI/automem/.venv/bin/pytest -q tests/test_backup_endpoint.py
\`\`\`

Expected: all backup/restore tests pass, including the optional HNSW configuration and local clone wrapper checks.

**Step 6: Commit the migration-tool change.**

\`\`\`bash
git add scripts/restore_from_backup.py scripts/lab/clone_production.sh tests/test_backup_endpoint.py
git commit -m "perf(qdrant): support disk-backed HNSW restores"
\`\`\`

---

### Task 3: Document the explicit production migration and operational controls

**Files:**
- Modify: \`docs/ENVIRONMENT_VARIABLES.md\`
- Modify: \`docs/MONITORING_AND_BACKUPS.md\`

**Step 1: Document the restore variable.**

Add \`QDRANT_RESTORE_HNSW_ON_DISK\` to the existing Qdrant restore tuning documentation in \`docs/ENVIRONMENT_VARIABLES.md\`. Describe it as an optional boolean that places the restored collection's HNSW index on disk; leave it unset by default so direct restore users must opt in.

**Step 2: Add a Qdrant-only migration runbook.**

Under the API backup export section in \`docs/MONITORING_AND_BACKUPS.md\`, add a concise “Migrate an existing Qdrant collection to on-disk storage” subsection that:

1. takes a \`?include=qdrant\` portable backup with the admin token;
2. records the pre-migration point count and takes a recoverable Railway volume backup/snapshot;
3. runs \`restore_from_backup.py\` against the intended Qdrant endpoint with \`--qdrant-only --force\` plus \`QDRANT_RESTORE_VECTOR_ON_DISK=true\`, \`QDRANT_RESTORE_HNSW_ON_DISK=true\`, and \`QDRANT_RESTORE_ON_DISK_PAYLOAD=true\`;
4. verifies point count and representative recall before calling the change complete; and
5. directs the operator to resize the Railway volume manually only after validation and a stable observation period.

State clearly that the restore deletes and recreates the selected Qdrant collection and that a tested backup is the rollback path. Do not document an automated Railway resize command.

**Step 3: Review documentation for credential and safety clarity.**

Run:

\`\`\`bash
rg -n "QDRANT_RESTORE_HNSW_ON_DISK|on-disk storage|qdrant-only|Railway volume" docs/ENVIRONMENT_VARIABLES.md docs/MONITORING_AND_BACKUPS.md
\`\`\`

Expected: the variable, destructive restore scope, validation requirements, and manual-resize boundary are each discoverable.

**Step 4: Commit the docs change.**

\`\`\`bash
git add docs/ENVIRONMENT_VARIABLES.md docs/MONITORING_AND_BACKUPS.md
git commit -m "docs(qdrant): add on-disk migration runbook"
\`\`\`

---

### Task 4: Verify the integrated change set

**Files:**
- Verify: all files changed by Tasks 1–3

**Step 1: Run formatting and lint checks.**

Run:

\`\`\`bash
make fmt
make lint
\`\`\`

Expected: Black/Isort produce no unintended changes and Flake8 passes.

**Step 2: Run the full unit suite.**

Run:

\`\`\`bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/jgarturo/Projects/OpenAI/automem/.venv/bin/pytest -q -m unit
\`\`\`

Expected: full unit suite passes.

**Step 3: Inspect the final diff and status.**

Run:

\`\`\`bash
git diff develop...HEAD --check
git status --short
git log --oneline develop..HEAD
\`\`\`

Expected: no whitespace errors, only intended files changed, and focused conventional commits are present.

**Step 4: Report the handoff boundary.**

State that the code provides defaults for new collections and an explicit migration mechanism for existing collections, but production migration and any Railway volume resize remain a manual operator action after backup and verification.
