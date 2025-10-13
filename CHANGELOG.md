# Changelog

All notable changes to AutoMem will be documented in this file.

## [Unreleased] - 2025-10-13

### 🎯 Major Data Quality Overhaul

#### Fixed - Memory Type Pollution (Critical)
- **Issue**: 490/773 memories (64%) had invalid types from recovery script bug
- **Root Cause**: Recovery script flattened `metadata.type` as top-level property, overwriting actual memory type
- **Fix**: 
  - Added `RESERVED_FIELDS` filter in `scripts/recover_from_qdrant.py` to exclude type, confidence, content, etc.
  - Created `scripts/cleanup_memory_types.py` to reclassify all 490 polluted memories
  - 100% success rate - all memories reclassified to 7 valid types
- **Impact**: Analytics now accurate with only valid memory types (Decision, Pattern, Preference, Style, Habit, Insight, Context)

#### Fixed - Tag-Based Search Not Working
- **Issue**: Tag searches returned 0 results despite memories having tags
- **Root Cause**: Qdrant missing keyword indexes on `tags` and `tag_prefixes` payload fields
- **Fix**: Added `PayloadSchemaType.KEYWORD` indexes in `_ensure_qdrant_collection()`
- **Impact**: Tag filtering now works for both exact and prefix matching

#### Improved - Entity Extraction Quality
**Phase 1 - Error Code Filtering:**
- Added `ENTITY_BLOCKLIST` for HTTP errors (Bad Request, Not Found, etc.)
- Added `ENTITY_BLOCKLIST` for network errors (ECONNRESET, ETIMEDOUT, etc.)

**Phase 2 - Code Artifact Filtering:**
- Block markdown formatting (`- **File:**`, `# Header`, etc.)
- Block class name suffixes (Adapter, Handler, Manager, Service, etc.)
- Block JSON field names (starting with underscore like `_embedded`)
- Block boolean literals (`'false'`, `true`, `null`)
- Block environment variables (all-caps with underscores)
- Block text fragments (strings ending with colons)

**Phase 3 - Project Name Extraction:**
- **Issue**: Project extraction regex required `Project ProjectName` but memories use `project: project-name`
- **Fix**: Added pattern `r"(?:in |on )?project:\s+([a-z][a-z0-9\-]+)"` 
- **Impact**: Now correctly extracts claude-automation-hub, mcp-automem, autochat from session starts

#### Added - LLM-Based Memory Classification
- **Feature**: Hybrid classification system (regex first, LLM fallback)
- **Implementation**: 
  - Enhanced `MemoryClassifier` with `_classify_with_llm()` method
  - Uses GPT-4o-mini for accurate, cost-effective classification
  - Falls back to LLM only when regex patterns don't match (~30% of cases)
- **Cost**: ~$0.03 one-time for 510 memories, ~$0.24/year ongoing (10 memories/day)
- **Accuracy**: 0.9 confidence on technical content, work logs, session starts
- **Script**: Created `scripts/reclassify_with_llm.py` for one-time reclassification

#### Added - Automated Backups
- **GitHub Actions**: Workflow runs every 6 hours, backs up to S3
- **Railway Volumes**: Built-in volume backups for redundancy  
- **Dual Storage**: FalkorDB (primary) + Qdrant (backup) provides data resilience
- **Documentation**: Consolidated all backup docs into `docs/MONITORING_AND_BACKUPS.md`

#### Fixed - FalkorDB Data Loss Recovery
- **Issue**: FalkorDB restarted and lost all memories (persistence not working)
- **Recovery**: Used `scripts/recover_from_qdrant.py` to restore 778/780 memories from Qdrant
- **Root Cause Analysis**: `appendonly` was set to `no` instead of `yes`
- **Prevention**: Verified `REDIS_ARGS` in Railway and documented proper persistence configuration

### 📁 Files Modified
- `app.py` - Enhanced entity extraction, LLM classification, Qdrant indexes
- `scripts/recover_from_qdrant.py` - Fixed metadata field handling
- `scripts/cleanup_memory_types.py` - Created tool for type reclassification
- `scripts/reclassify_with_llm.py` - Created LLM-based reclassification tool
- `docs/MONITORING_AND_BACKUPS.md` - Updated with comprehensive backup strategies
- `docs/RAILWAY_DEPLOYMENT.md` - Updated references to backup documentation
- `README.md` - Updated architecture and documentation links
- `.github/workflows/backup.yml` - Created automated backup workflow

### 📊 Impact Summary
- **Memory Types**: 79 → 7 valid types (92% reduction in pollution)
- **Reclassified**: 490 memories (100% success rate)
- **Entity Quality**: Clean extraction, no error codes or junk
- **Tag Search**: Fully functional with proper indexing
- **Backups**: Automated every 6 hours to S3
- **Data Recovery**: 99.7% recovery rate (778/780) from Qdrant
- **Classification**: Hybrid system with 0.9 LLM confidence

### 🚀 Deployment
All changes deployed to Railway at `automem.up.railway.app`

### 💰 Cost Analysis
- **LLM Classification**: $0.03 one-time + $0.24/year ongoing
- **S3 Backups**: Included in existing AWS usage
- **Qdrant**: Existing free tier
- **Railway**: Existing deployment

---

## [Prior Releases]
See git history for previous changes.
