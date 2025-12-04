# Memory System Competitor Analysis
## Why CORE Gets 88% and How AutoMem Can Close the Gap

*Analysis Date: December 3, 2025*

---

## Executive Summary

After analyzing CORE, mem0, and Zep/Graphiti architectures, I identified **5 key techniques** that could close the 12.5% gap:

| Technique | CORE | AutoMem | Impact |
|-----------|------|---------|--------|
| Statement-Level Facts | ✅ | ❌ | High |
| Temporal Provenance | ✅ | Partial | Medium |
| Contradiction Resolution | ✅ | ❌ | Medium |
| Multi-Angle Search | ✅ | ✅ | Already have |
| Re-ranking with Diversity | ✅ | Partial | Medium |

**Bottom Line**: AutoMem's core retrieval is solid (we do vector + graph). The gap is in **fact extraction** and **temporal reasoning**.

---

## CORE's Architecture (88.24%)

### What Makes CORE Different

CORE uses a **reified temporal knowledge graph** with a 4-phase ingestion pipeline:

```
Episode → Normalization → Extraction → Resolution → Graph Integration
```

#### 1. Statement-Level Facts (HIGH IMPACT)

CORE doesn't just extract entities. It extracts **statements with full provenance**:

```
Input: "Emma mentioned she works as a software engineer at Google"

CORE extracts:
├── Entities: [Emma, software engineer, Google]
├── Statement: "Emma works as a software engineer at Google"
├── Relationship: "works_as_at"
├── Source: dialog_id_42
├── Timestamp: 2024-03-15
└── Confidence: 0.95
```

**AutoMem currently**: Stores the raw text as a memory unit. Extracts entities, but doesn't create first-class "fact" objects.

**Recommendation**: Add a `facts` table that extracts declarative statements from memories.

#### 2. Temporal Provenance Tracking (MEDIUM IMPACT)

CORE tracks:
- **When** something was said (event time)
- **When** it was ingested (system time)
- **Who** said it (speaker attribution)
- **How** it evolved (superseded_by relationships)

```
Fact: "Project deadline is March 15"
├── valid_from: 2024-01-10
├── valid_until: 2024-02-20 (superseded)
├── superseded_by: "Project deadline is April 1"
└── speaker: "Alice"
```

**AutoMem currently**: Has `t_valid`/`t_invalid` fields but doesn't use them for supersession.

**Recommendation**: Implement automatic fact supersession when conflicting info is detected.

#### 3. Contradiction Resolution (MEDIUM IMPACT)

CORE explicitly handles contradictions:
- Detects when new facts conflict with old ones
- Creates "INVALIDATED_BY" relationships instead of overwriting
- Preserves full history so you can answer "What did we originally think?"

**AutoMem currently**: Has `INVALIDATED_BY` relationship type but no automatic detection.

**Recommendation**: Add LLM-based contradiction detection during enrichment.

#### 4. Multi-Angle Search (ALREADY HAVE ✅)

CORE searches from multiple angles simultaneously:
- Keyword search (exact matches)
- Semantic search (embedding similarity)
- Graph traversal (connected facts)

**AutoMem already does this!** This is our strength.

#### 5. Re-ranking with Diversity (PARTIAL)

CORE's re-ranking:
- Doesn't just return most similar results
- Ensures diversity of perspectives
- Prioritizes recent + reliable + connected facts

**AutoMem partially has this** with graph distance weighting, but could add diversity scoring.

---

## Mem0's Architecture (Claims +26% over OpenAI Memory)

### Key Differentiators

1. **Memory Compression Engine**: Automatically summarizes and compresses long-term memories
2. **Multi-Level Memory**: User, Session, and Agent state tracked separately
3. **Self-Improving**: Learns which memories are useful based on retrieval patterns

### What We Could Borrow

- **Memory compression**: We don't do this. Could help with token efficiency.
- **Retrieval feedback loop**: Track which memories get used and boost them.

---

## Zep/Graphiti Architecture (Published in arXiv)

### Key Differentiators

1. **Bi-Temporal Data Model**: Explicit event time vs. ingestion time
2. **Hybrid Retrieval**: BM25 + semantic + graph traversal (sub-second latency)
3. **Custom Entity Definitions**: Pydantic models for domain-specific entities
4. **Community Detection**: Groups related facts into "communities"

### What We Could Borrow

- **BM25 for keyword search**: We use Qdrant's keyword search but BM25 might be better
- **Community detection**: Could help with multi-hop reasoning

---

## Specific Recommendations for AutoMem

### 1. Add Statement Extraction (HIGH PRIORITY)

```python
# Current enrichment extracts entities and topics
# Add: Extract declarative statements

class Fact(BaseModel):
    statement: str        # "Emma works as a software engineer"
    subject: str          # "Emma"
    predicate: str        # "works as"
    object: str           # "software engineer"
    source_memory_id: str
    valid_from: datetime
    valid_until: Optional[datetime]
    confidence: float
```

### 2. Implement Automatic Contradiction Detection (MEDIUM PRIORITY)

During enrichment, check if new facts contradict existing ones:

```python
async def detect_contradictions(new_fact: Fact, user_id: str) -> List[Fact]:
    """Find existing facts that conflict with new fact."""
    # Search for facts about the same subject
    existing = await recall_facts_about(new_fact.subject, user_id)
    
    # Use LLM to detect contradictions
    for old_fact in existing:
        if llm_detects_contradiction(old_fact, new_fact):
            # Mark old fact as superseded
            old_fact.valid_until = new_fact.valid_from
            old_fact.superseded_by = new_fact.id
```

### 3. Add Temporal Reasoning to Recall (MEDIUM PRIORITY)

When answering questions like "What was the deadline?", consider:
- Current valid facts (default)
- Historical facts if asking "originally" or "before"
- Most recent fact wins for conflicts

### 4. Response Time Advantage (KEEP)

AutoMem: ~70ms
CORE: ~600ms

**Don't sacrifice this!** Our speed is a competitive advantage.

---

## Why This Matters for LoCoMo Categories

| Category | Gap | Root Cause | Fix |
|----------|-----|------------|-----|
| Single-hop (66% vs 91%) | 25% | Statement extraction | Add facts table |
| Temporal (51% vs 88%) | 37% | No temporal reasoning | Add supersession |
| Multi-hop (35% vs 85%) | 50% | Can't chain facts | Statement extraction + graph |
| Open Domain (84% vs 71%) | -13% | **We're ahead!** | Keep as-is |

**Multi-hop** is where we lose the most. This is because multi-hop requires:
1. Finding fact A: "Emma works at Google"
2. Finding fact B: "Google is headquartered in Mountain View"
3. Combining: "Emma works in Mountain View"

Without statement-level facts, we can't do this reasoning reliably.

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 weeks)
1. Add diversity scoring to re-ranking
2. Implement temporal filtering in recall (use t_valid/t_invalid)
3. Better prompt engineering for answer generation

### Phase 2: Statement Extraction (2-4 weeks)
1. Add `facts` extraction to enrichment pipeline
2. Store as separate graph nodes with `STATES` relationship
3. Enable fact-based recall alongside memory recall

### Phase 3: Contradiction Resolution (1-2 weeks)
1. LLM-based contradiction detection during enrichment
2. Automatic supersession of conflicting facts
3. Historical fact queries

---

## Conclusion

The 12.5% gap is real but closable. Key insight: **AutoMem's retrieval is good; our fact extraction is the weakness.**

CORE's advantage is not magic - it's a more sophisticated representation of knowledge. By adding statement-level facts and temporal reasoning, we could realistically achieve 80-85% on LoCoMo while maintaining our ~10x speed advantage.

**Recommended Claims**:
- "75.73% on LoCoMo with CORE-comparable evaluation"
- "10x faster response times than CORE"
- "91.7% on adversarial questions (knowing what you don't know)"
- "Closing the gap through statement-level fact extraction (roadmap)"

---

*References*:
- CORE: https://github.com/RedPlanetHQ/core
- Mem0: https://github.com/mem0ai/mem0
- Zep/Graphiti: https://arxiv.org/abs/2501.13956

