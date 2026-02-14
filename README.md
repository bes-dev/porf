# PORF

**Publish Original Research Fast**

PORF is a minimalist Python library that writes research articles on any topic. It runs a multi-expert roundtable discussion, collects sources, and produces a polished long-form article with citations.

Named after Porfiry Petrovich — the literary AI from Victor Pelevin's *iPhuck 10*.

## Installation

```bash
pip install porf

# With interactive CLI (recommended)
pip install porf[cli]

# With DuckDuckGo search (free, no API key)
pip install porf[search]

# With semantic retrieval (better source routing)
pip install porf[semantic]

# Everything
pip install porf[all]
```

## Quick Start

### CLI

```bash
# Interactive wizard
porf

# One-liner
porf "Impact of AI on creative industries" -p deep -t popular -o report.md

# Custom article length
porf "AI in healthcare" -p balanced -w 5000 -o report.md

# Multi-language search, output in Russian
porf "AI in healthcare" -l en,ru --out-lang ru -o report.md
```

### Python API

```python
from porf import research

report = research(
    topic="The Impact of Large Language Models on Software Development",
    profile="balanced",
    style="analytical",
    target_words=5000,
)

print(report.markdown)
```

## How It Works

PORF follows a four-stage pipeline:

```
1. EXPLORE
   Decompose topic → search queries → collect initial findings

2. BOOTSTRAP
   Generate expert panel with clashing perspectives
   Design article outline (sections with claims, optional subsections)

3. ROUNDTABLE (multi-round)
   Each expert researches their assigned section:
     generate queries → search → draft with citations
   Information routed to mind map via embeddings
   Sections auto-lock when they converge
   Sections can split into subsections during research

4. WRITE
   Narrative plan → write body sections (sequential) →
   write intro → write conclusion → assemble with hierarchy
```

The roundtable produces research drafts with citations. The writer transforms them into a polished article with narrative flow, style-appropriate prose, and consistent citation numbering.

## Profiles & Styles

**Profiles** control research depth and article length:

| Profile | Experts | Max rounds | Lock threshold | Target words |
|---------|---------|------------|----------------|--------------|
| `quick` | 3 | 10 | 5 sources | ~2000 |
| `balanced` | 4 | 15 | 8 sources | ~4000 |
| `deep` | 5 | 20 | 12 sources | ~6000 |

**Styles** control article tone and writing techniques:

| Style | Tone |
|-------|------|
| `analytical` | Systematic, fair, data-driven |
| `academic` | Precise, authoritative, theory-friendly |
| `journalistic` | Direct, concrete, fact-first |
| `popular` | Conversational, accessible, theory-light |
| `essay` | Reflective, personal, exploratory |

```bash
porf "Quantum computing" -p deep -t journalistic
```

## CLI Reference

```
porf [TOPIC] [OPTIONS]

Arguments:
  TOPIC                    Research topic (wizard mode if omitted)

Options:
  -p, --profile PROFILE    quick, balanced, deep (default: balanced)
  -t, --style STYLE        analytical, academic, journalistic, popular, essay
  -w, --words N            Target article length in words (overrides profile)
  -m, --model MODEL        LLM model (default: anthropic/claude-sonnet-4-20250514)
  -s, --search ENGINE      duckduckgo, tavily, brave, serper, searxng
  -l, --lang LANG          Search language(s): auto, en, ru, or en,ru
  --out-lang LANG          Output language: auto, en, ru, etc.
  -o, --output FILE        Output file path
  --api-base URL           API base URL for local models
```

## Search Engines

| Engine | API Key | Notes |
|--------|---------|-------|
| `duckduckgo` | Not needed | Default, free |
| `tavily` | `TAVILY_API_KEY` | Best for research |
| `brave` | `BRAVE_API_KEY` | Good quality |
| `serper` | `SERPER_API_KEY` | Google results |
| `searxng` | Not needed | Self-hosted |

## Local Models

```python
from porf import research

# LM Studio
report = research(
    topic="...",
    profile="quick",
    model="openai/local-model",
    api_base="http://localhost:1234/v1",
)

# Ollama
report = research(
    topic="...",
    model="ollama/llama3",
    api_base="http://localhost:11434",
)
```

## Output

```python
report.markdown   # Full article with citations
report.sections   # List of Section objects
report.sources    # List of Source objects
report.to_dict()  # JSON-serializable dict
```

## License

Apache 2.0
