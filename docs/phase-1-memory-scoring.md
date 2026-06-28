# Phase 1 — Memory Section Scoring

> **Prerequisite:** Phase 0 (rename and restructure) must be done first.
> After Phase 0, the package is `jiuwenswarm/tools/recommendation_matrix/` with a
> `shared/` folder containing all the generic pipeline code.
>
> Phase 1 fills in `recommendation_matrix/memory/` — two new files, using shared code.

---

## The problem with memory today

When the agent runs, it injects the full contents of:
- `<project>/.jiuwenswarm/project.md` — project-specific notes
- `~/.jiuwenswarm/user.md` — user preferences and personal notes

These files are injected whole, every time, regardless of the question.

A real `project.md` might have sections like:

```
## Architecture overview
## Known bugs
## Deployment setup
## Team contacts
## Sales performance targets
## Onboarding notes
```

If the user asks "fix the null pointer error in the parser", the architecture and bug
sections are relevant. The sales targets and onboarding notes are not. Injecting all of
them wastes tokens and may confuse the response.

Memory sections are exactly like skills in this respect: some are relevant to a query,
some are not, and relevance can be pre-computed. The same pipeline that scores skills
scores memory sections with minimal additional code.

---

## Where the new code lives

After Phase 0, the directory structure is:

```
jiuwenswarm/tools/recommendation_matrix/
  shared/                    ← all generic pipeline code (from Phase 0)
  skills/                    ← skill-specific scanner + composer (from Phase 0)
  memory/                    ← Phase 1 adds these two files:
    __init__.py
    scanner.py               ← MemorySectionScanner
    composer.py              ← MemoryMatrixComposer
  tools/                     ← empty, ready for Phase 2
```

**That is all that Phase 1 adds.** Two new files in `memory/`. Everything else — the
LLM calls, the scoring, the matrix file writing, the fingerprinting, the state
management — comes from `shared/` without modification.

---

## `memory/scanner.py` — what it does

`MemorySectionScanner` reads `project.md` and `user.md` and splits each file at every
`##` heading. Each section becomes one `ComponentRecord` (the same dataclass used for
skills in `shared/fingerprint.py`).

```
project.md content:

## Architecture overview
The system uses a microservices layout...

## Known bugs
- Null pointer in parser when input is empty
```

Becomes two `ComponentRecord` objects:

```python
ComponentRecord(
    name="Architecture overview",
    description="The system uses a microservices layout",  # first sentence
    body="The system uses a microservices layout...",      # full section text
    mtime=1697276400.0,  # mtime of project.md
    directory=Path("~/.jiuwenswarm"),
)

ComponentRecord(
    name="Known bugs",
    description="- Null pointer in parser when input is empty",
    body="- Null pointer in parser when input is empty...",
    mtime=1697276400.0,
    directory=Path("~/.jiuwenswarm"),
)
```

The `ComponentRecord` fields are identical to what skills use. This is why the shared
pipeline accepts it without any changes.

**Filtering rules inside the scanner:**
- Sections with fewer than 50 characters of body text are skipped (too short)
- Sections with heading level `###` or deeper are skipped (treat only `##` as sections)
- The first line of each file (if it is a `#` title) is skipped — it is the file name,
  not a section

**Fingerprinting:** SHA256 of `(name + body + mtime of source file)` — same formula
as `skill_fingerprint()` in `shared/fingerprint.py`. A section is considered changed
if either its text changed or the source file was modified.

---

## `memory/composer.py` — what it does

`MemoryMatrixComposer` is the orchestrator for memory section scoring. It is structured
identically to `skills/composer.py` (which itself is the renamed `matrix_composer.py`
from the old `skill_matrix/`).

```python
class MemoryMatrixComposer:
    def __init__(self, project_dir, matrix_dir, model, model_name, n_examples, parallel):
        self._scanner    = MemorySectionScanner(project_dir)
        self._determiner = ChangedItemsDeterminer(matrix_dir, state_file="matrix_state_memory.json")
        self._generator  = QueriesGenerator(model, model_name, n_examples, parallel,
                                            prompt_template=_MEMORY_PROMPT_TEMPLATE)
        self._evaluator  = AllItemsEvaluator(model, model_name, matrix_dir, parallel,
                                             file_prefix="mem")
        self._saver      = StateSaver(matrix_dir, state_file="matrix_state_memory.json")

    async def build(self, force=False, only=None):
        sections  = self._scanner.scan(only)
        changed, skipped = self._determiner.determine(sections, force)
        gen_results = await self._generator.generate_for_items(changed)
        states, llm_calls = await self._evaluator.evaluate_all(gen_results)
        self._saver.save(sections, skipped, states)
```

The only two things that differ from `SkillMatrixComposer`:
1. The scanner (`MemorySectionScanner` instead of `ExistingSkillsScanner`)
2. The query generation prompt template (see below)

Everything else — the LLM calls, the scoring, the file writing, the state — is the
same shared code.

---

## The query generation prompt

`QueriesGenerator` in `shared/` already accepts a `prompt_template` argument.
Phase 1 passes a different template that asks the right question for memory sections:

```
You are given a section from a project memory file.
Generate {n} realistic user queries that this section would help answer.

Each example must have:
  - "query": a natural user message (1-3 sentences) that would benefit from knowing
    the information in this section
  - "answer": what a good response looks like when this information is available
    (1-2 sentences, concrete)

Source file: {source_file}
Section heading: {name}
First sentence: {description}

Section content:
{body}

Requirements:
- Queries must be realistic user questions or task requests, not descriptions of the section.
- Cover the different facts or sub-topics within this section.
- Do not generate queries for information that is not in this section.

Return a JSON array of exactly {n} objects:
[{{"query": "...", "answer": "..."}}, ...]
No other text.
```

The template uses the same `{name}`, `{description}`, `{body}` placeholders as the
skill template. `{source_file}` is an extra field that `ComponentRecord` carries for
memory sections.

---

## Output files

Memory matrix files go into the **same `oracle/` directory** as skill matrix files.
They are distinguished by a `mem_` prefix in the filename.

```
~/.jiuwenswarm/agent/workspace/oracle/
  scoring_matrix_skill_email_sender.json    ← skills (Phase 0 renamed these)
  scoring_matrix_skill_devops_toolkit.json
  scoring_matrix_mem_architecture_overview.json   ← memory (Phase 1)
  scoring_matrix_mem_known_bugs.json
  scoring_matrix_mem_deployment_setup.json
  matrix_state_skills.json
  matrix_state_memory.json                        ← memory state (Phase 1)
```

File format is identical to skill matrix files. One extra field added:

```json
{
  "run_id": "mem_architecture_overview_20251014T090000",
  "component_type": "memory_section",
  "component_name": "Architecture overview",
  "source_file": "project.md",
  "fitness_metrics": ["f1", "bigram_f1", "bag_of_words", "length_ratio"],
  "baseline_cross_eval": [
    {
      "example_id": "mem_architecture_overview_0000",
      "example_input": "How is the backend structured?",
      "example_expected": "The system uses microservices...",
      "candidate_output": "Based on the architecture notes, the system...",
      "scores": { "f1": 0.71, "bigram_f1": 0.65, "bag_of_words": 0.68, "length_ratio": 0.91 }
    }
  ],
  "evolved_cross_eval": []
}
```

The `component_type` and `source_file` fields are new. The rest is identical.

---

## CLI

The unified CLI from Phase 0 is used:

```bash
# Build only memory matrix
python -m jiuwenswarm.tools.recommendation_matrix build --type memory \
    --project-dir ~/.jiuwenswarm \
    --matrix-dir ~/.jiuwenswarm/agent/workspace/oracle \
    --model gpt-4o-mini \
    --api-key $OPENAI_API_KEY \
    --n-examples 15

# Build everything (skills + memory) in one command
python -m jiuwenswarm.tools.recommendation_matrix build \
    --project-dir ~/.jiuwenswarm \
    --skills-dir ~/.jiuwenswarm/skills \
    --matrix-dir ~/.jiuwenswarm/agent/workspace/oracle \
    --model gpt-4o-mini \
    --api-key $OPENAI_API_KEY
```

No separate CLI for memory. The `--type memory` flag routes to `MemoryMatrixComposer`.

---

## How selection works at query time

The existing `RecommendSkillTool` is updated to also load `scoring_matrix_mem_*.json`
files and treat them as selectable memory sections. Alternatively, a separate
`RecommendMemoryTool` is created (parallel structure).

The recommender uses the same TF-IDF cosine similarity it uses for skills:
1. Fit TF-IDF on all `example_input` texts from all memory matrix files
2. At query time: cosine similarity of user query → find top-matching sections
3. Include sections whose nearest example exceeds the similarity threshold

**Integration point:** `ProjectMemoryRail`
(`agents/harness/common/rails/project_memory/`) gains a `recommendation_mode` flag.
When set to `True`, it calls the memory recommender instead of loading full files.

---

## LLM cost estimate

Assuming `project.md` has 8 sections and `user.md` has 4, `n_examples=15`:

| Step | Calls |
|---|---|
| Query generation (Stage 1) | 12 sections × 1 = **12 calls** |
| Evaluation (Stage 2) | 12 sections × 15 queries = **180 calls** |
| Total | **192 LLM calls** |

Rebuild is triggered only when `project.md` or `user.md` changes
(fingerprint check in `ChangedItemsDeterminer`).

---

## Implementation checklist

| Task | File | Shared code reused |
|---|---|---|
| Memory section scanner | `memory/scanner.py` | Uses `ComponentRecord` from `shared/fingerprint.py` |
| Query prompt template | `memory/composer.py` | Passed to `QueriesGenerator` in `shared/` |
| Composer orchestrator | `memory/composer.py` | Uses all of `shared/`: generator, evaluator, determiner, saver |
| `--type memory` in CLI | `cli.py` | Already structured to accept type |
| Output file naming | `shared/all_items_evaluator.py` | `file_prefix="mem"` parameter |
| State file | `shared/state_saver.py` | `state_file="matrix_state_memory.json"` parameter |
| `ProjectMemoryRail` update | `agents/harness/common/rails/project_memory/` | Adds `recommendation_mode` option |

**Total new code in `memory/`: ~120 lines.**
Everything else is parameter changes to existing shared code.

---

## What Phase 1 unlocks

- Memory injection becomes precise: only relevant sections reach the LLM
- Context size reduced by 40–70% in typical cases without losing useful information
- Sections that consistently score low can be flagged as candidates to remove or rewrite
- Foundation for Phase 3: memory section scores become inputs to the evolutionary search
