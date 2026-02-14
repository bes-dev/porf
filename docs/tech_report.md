# PORF: Multi-Expert Roundtable for Automated Deep Research

**Technical Report v0.3**

## Abstract

PORF (Publish Original Research Fast) is a minimalist framework for automated deep research. It combines a multi-expert roundtable discourse (inspired by Stanford's Co-STORM) with a dedicated writer that produces polished long-form articles. Key design choices: (1) a mind map knowledge structure with embedding-based information routing, (2) section-level convergence with adaptive thresholds, (3) a sequential writer with narrative planning and anti-pattern enforcement, and (4) language-independent style control. The system produces articles with hierarchical structure, global citation numbering, and style-appropriate prose in ~1000 lines of core logic.

## 1. Introduction

Automated research systems face a fundamental tension: research quality requires breadth and depth, while readable output requires narrative coherence and stylistic control. Most systems treat these as a single problem — the LLM both researches and writes. PORF separates them: a multi-expert roundtable handles research, and a dedicated writer transforms the results into prose.

This separation yields two benefits. First, the roundtable can focus on finding and organizing evidence without worrying about style. Second, the writer can focus on narrative flow without being distracted by search queries. Each component does one thing well.

## 2. Related Work

### 2.1 Stanford STORM / Co-STORM

STORM [1] introduced multi-perspective question generation for Wikipedia-style articles. Co-STORM [6] extended this with a mind map knowledge structure and collaborative discourse between experts. PORF builds directly on Co-STORM's architecture — the mind map, expert personas, and iterative research rounds — while adding a dedicated writing phase and section-level convergence.

### 2.2 LongWriter / AgentWrite

LongWriter [7] identified that LLMs have an output length ceiling determined by training data distribution. AgentWrite addresses this by planning paragraphs, then writing sequentially with growing context. PORF's writer uses a similar sequential approach (each section sees the previous section's ending) but avoids explicit paragraph planning — section-level granularity is sufficient when sections are properly decomposed.

### 2.3 OpenAI Deep Research

OpenAI's Deep Research [2] uses end-to-end RL on browsing tasks with a Plan-Act-Observe loop. PORF takes a simpler approach: structured prompts with predefined research profiles, no RL training required.

## 3. Architecture

### 3.1 Pipeline Overview

```
Input: Topic + Profile + Style
    ↓
Phase 1: EXPLORE
    Decompose topic → search queries → initial findings
    ↓
Phase 2: BOOTSTRAP
    Generate expert panel (clashing perspectives)
    Design article outline (sections with claims, optional subsections)
    ↓
Phase 3: ROUNDTABLE (multi-round)
    For each round:
      Assign each expert to the least-explored unlocked section
      Expert: generate queries → search → filter → draft section
      Route cited sources to mind map nodes (embedding + LLM)
      Check convergence → auto-lock sections with enough sources
    ↓
Phase 4: WRITE
    4a. Build global source index (deduplicated, numbered)
    4b. Narrative plan (thread, hook, conclusion angle, section order)
    4c. Write body sections (sequential, each sees previous ending)
    4d. Write intro (after body — journalistic order)
    4e. Write conclusion (synthesis, not summary)
    4f. Assemble with tree-aware header levels
    ↓
Output: Report with hierarchical sections + global citations
```

### 3.2 Mind Map Knowledge Structure

The mind map is a tree rooted at the topic. Each node represents a section or subsection and accumulates:

- **snippet_uuids**: set of source references attached to this node
- **claim**: the section's central argument
- **draft**: research draft with local citations
- **open_questions**: what still needs investigation
- **contributors**: which experts have worked on this section

Snippets are routed to nodes using a two-stage process:
1. **Embedding ranking**: encode query + snippet, compute cosine similarity against all node paths, select top-8 candidates
2. **LLM selection**: ask the model which candidate is the best fit

This hybrid approach is fast (embeddings narrow the search) and accurate (LLM makes the final decision).

### 3.3 Section Convergence

Sections auto-lock when they have enough sources AND a draft. The threshold depends on depth:

- Top-level sections: lock at `profile.lock_sources` (e.g. 12 for deep)
- Subsections (depth > 1): lock at `lock_sources // depth` (min 3)

This allows subsections to converge faster — they cover narrower topics and need fewer sources. Once all sections are locked, the roundtable ends early.

### 3.4 Section Granularity

PORF encourages natural section decomposition at two points:

1. **BOOTSTRAP**: the initial outline can include subsections when a section covers multiple distinct aspects
2. **DRAFT_SECTION**: during research, experts can propose restructuring (split, merge, rename, add, drop)

This produces finer-grained write units (~400 words each), which helps the writer hit target word counts — LLMs generate more total text when writing many short sections than few long ones.

### 3.5 Writer

The writer transforms research drafts into a polished article in 5 phases:

**Phase 1: PREPARE.** Build a global source index (deduplicated by normalized URL) and compute per-section word budgets proportional to source count.

**Phase 2: NARRATIVE_PLAN.** One LLM call receives all top-level section names, claims, and draft previews. It produces: a narrative thread (the "red thread" connecting sections), an intro hook (concrete scene or fact), a conclusion angle (synthesis, not summary), and optionally reorders top-level sections for better narrative flow.

**Phase 3: WRITE_SECTION.** Sequential — each section sees the ending of the previous section for narrative continuity. Each call receives: the research draft (with citations remapped to global numbers), the section's sources (with global numbering), style techniques and tone, anti-patterns to avoid, mandatory prose techniques, a list of already-covered topics (to prevent duplication), and a target word count.

**Phase 4: WRITE_INTRO + WRITE_CONCLUSION.** The intro is written AFTER all body sections (journalistic practice — the lede is written last). It receives the first section's beginning to avoid duplication. Overlap trimming removes any bleed. The conclusion receives all section claims and synthesizes without retelling.

**Phase 5: ASSEMBLE.** Walk the section tree recursively, preserving hierarchy: top-level sections become `##`, subsections become `###`, deeper nesting gets deeper levels. Parent nodes with children become header-only sections.

### 3.6 Style Control

Instead of few-shot examples (language-dependent), PORF uses structural descriptions:

**Style techniques** (per style): describe HOW to write — what to open sections with, how to handle theory, how to end sections. Example for `popular`:

> Open sections with a vivid scene, specific person, or surprising fact. If the research material references a theorist or theory, explain the idea in plain language first — the reader shouldn't need prior knowledge. One theoretical reference per section maximum.

**Anti-patterns** (universal): describe patterns to AVOID, in language-independent terms:

> Inflated importance, mechanical transitions ("Furthermore"), hedging ("It's important to note"), hollow filler gerunds, monotonous section openings (always echoing previous ending with "but...").

**Prose techniques** (universal): mandatory structural techniques:

> Vary sentence length dramatically. Use contractions. Open with concrete scene, never thesis statement. Transition between sections through narrative flow, never use the same transition technique twice in a row.

The model applies these descriptions to whatever output language is specified.

## 4. Research Profiles

| Parameter | quick | balanced | deep |
|-----------|-------|----------|------|
| Expert perspectives | 3 | 4 | 5 |
| Max rounds | 10 | 15 | 20 |
| Results per query | 3 | 5 | 5 |
| Queries per turn | 2 | 2 | 2 |
| Lock threshold (sources) | 5 | 8 | 12 |
| Target words | 2000 | 4000 | 6000 |

## 5. Implementation

### 5.1 File Structure

```
porf/
├── core.py       Main pipeline: research(), roundtable, writer
├── prompts.py    All LLM prompts (DECOMPOSE, BOOTSTRAP, RESEARCH_TURN,
│                 DRAFT_SECTION, NARRATIVE_PLAN, WRITE_SECTION, etc.)
├── mind_map.py   Tree structure with snippet accumulation
├── encoder.py    Embedding-based routing (cosine similarity)
├── search.py     Search engine adapters (DuckDuckGo, Tavily, Brave, etc.)
├── types.py      Data structures (Report, Section, Source)
└── cli.py        Interactive CLI with wizard mode
```

### 5.2 LLM Integration

PORF uses litellm for access to 100+ LLM providers. All LLM calls use `temperature=0.9`. Writer calls for body sections, intro, and conclusion output raw text (not JSON) to avoid escaping artifacts in prose.

### 5.3 Source Quality Control

Three layers of source filtering:

1. **Domain blocking**: social media, e-commerce, video platforms removed
2. **Domain diversity**: max 2 results per domain per query
3. **LLM relevance scoring**: each source rated 0-3, only score >= 2 kept

### 5.4 Citation Pipeline

1. During roundtable: experts cite sources as local `[1], [2]` within their drafts
2. At write time: `_build_source_index()` creates a global deduplicated source list
3. `_remap_citations()` converts local numbers to global numbers per section
4. Writer receives sources with global numbering and cites as `[N]` inline
5. Final article has consistent global citation numbers across all sections

## 6. Evaluation

### 6.1 Iterative Improvements

Tested on the topic "Чем хуже, тем лучше: как и почему современная мода превратилась в культурный мем" across three runs:

| Metric | Run 1 (Sonnet, balanced) | Run 2 (Opus, deep) | Run 3 (Opus 4.6, deep) |
|--------|--------------------------|---------------------|------------------------|
| Body sections | 7 | 11 | 15 |
| Word count (% of target) | 67% | 78% | 89% |
| Sources cited | ~30 | 78 | 134 |
| Intro duplication | Yes | Fixed | Fixed |
| Section ending variety | 1/7 types | 3/11 | 12/15 |
| Content duplication | N/A | High | Low (motif-level) |

### 6.2 Key Findings

- **Section granularity drives word count**: more fine-grained sections → more writer calls → closer to target length. LLMs reliably produce ~400 words per call but struggle with 800+.
- **Sequential writing with context creates flow**: passing the previous section's ending produces natural narrative transitions between sections.
- **Anti-patterns work**: language-independent pattern descriptions effectively reduce AI-detectable writing patterns (mechanical transitions, hedging, repetitive endings).
- **covered_topics prevents duplication**: accumulating a list of already-covered claims reduces content repetition between sections.
- **Subsection convergence needs lower thresholds**: subsections cover narrower topics; applying the same lock threshold as top-level sections wastes research rounds.

## 7. Limitations

- **No multimodal support**: text only
- **Quality depends on search engine**: limited by search coverage and snippet quality
- **No human-in-the-loop**: no interactive refinement during research
- **No fact verification**: citations indicate source, not correctness
- **Word count ceiling**: ~90% of target achievable; closing the gap requires architectural changes to LLM output length

## 8. API Reference

```python
def research(
    topic: str,
    model: str = "anthropic/claude-sonnet-4-20250514",
    embedding_model: str = "text-embedding-3-small",
    api_base: str | None = None,
    search: str | SearchEngine = "duckduckgo",
    search_api_key: str | None = None,
    profile: str = "balanced",       # "quick", "balanced", "deep"
    style: str = "analytical",       # "analytical", "academic", "journalistic",
                                     # "popular", "essay"
    search_languages: str | list[str] = "auto",
    output_language: str = "auto",
    target_words: int | None = None, # Override profile's target
    prompts_override: dict[str, str] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> Report
```

## References

[1] Shao, Y., et al. "Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models." NAACL 2024.

[2] OpenAI. "Introducing Deep Research." 2025.

[3] Various authors. "Deep Research Agents: A Systematic Examination and Roadmap." arXiv:2506.18096, 2025.

[6] Jiang, Y., et al. "Into the Unknown Unknowns: Engaged Human Learning through Participation in Language Model Agent Conversations." arXiv:2408.15232, 2024.

[7] Bai, Y., et al. "LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs." arXiv:2408.07055, 2024.
