# Phase 0 — Rename and Restructure (Prerequisite)

> **Must be done before Phase 1 or Phase 2.**
> This is not a new feature — it is a cleanup that makes Phases 1 and 2 possible
> without creating duplicated or messy code.

---

## The problem with the current layout

The current code lives in `jiuwenswarm/tools/skill_matrix/`. This was the right name
when only skills needed a matrix. Now that we are adding memory sections (Phase 1)
and tools (Phase 2), the name is wrong and the structure will cause problems.

If we add Phases 1 and 2 without restructuring, we end up with:

```
jiuwenswarm/tools/
  skill_matrix/          ← existing
    fingerprint.py
    state.py
    queries_generator.py       ← generic, duplicated in memory_matrix
    single_evaluator.py        ← generic, duplicated in memory_matrix
    ...

  memory_matrix/         ← Phase 1, new
    fingerprint.py             ← copy of skill_matrix version
    state.py                   ← copy of skill_matrix version
    queries_generator.py       ← copy of skill_matrix version
    single_evaluator.py        ← copy of skill_matrix version
    ...

  tool_matrix/           ← Phase 2, new
    fingerprint.py             ← third copy
    state.py                   ← third copy
    queries_generator.py       ← third copy
    ...
```

Three separate packages with three copies of the same pipeline code.
When a bug is found in `queries_generator.py`, it needs to be fixed in three places.
When a new metric is added, it must be added in three places.
This is the kind of structure that turns into a maintenance problem fast.

---

## The solution: one package, shared infrastructure

Rename `skill_matrix/` to `recommendation_matrix/` and restructure its internals
so that the generic pipeline lives in a `shared/` subfolder, and each component type
(skills, memory, tools) only adds what is unique to it.

```
jiuwenswarm/tools/
  recommendation_matrix/        ← renamed from skill_matrix
    __init__.py
    __main__.py
    cli.py                      ← single CLI for all component types

    shared/                     ← generic pipeline (moved from skill_matrix root)
      __init__.py
      fingerprint.py            ← ComponentRecord, fingerprinting functions
      state.py                  ← MatrixState, ComponentState dataclasses
      queries_generator.py      ← QueriesGenerator (unchanged logic)
      single_evaluator.py       ← SingleEvaluator (was SingleSkillEvaluator)
      all_items_evaluator.py    ← AllItemsEvaluator (was AllSkillsEvaluator)
      changed_items_determiner.py  ← unchanged
      state_saver.py            ← StateSaver (generalized)
      summary_printer.py        ← unchanged
      metrics/                  ← all metrics files, entirely unchanged
        __init__.py
        _tokenizer.py
        metric_token_f1.py
        metric_bigram_f1.py
        metric_bag_of_words.py
        metric_length_ratio.py
        metrics_list.py

    skills/                     ← skill-specific code (moved from skill_matrix root)
      __init__.py
      scanner.py                ← ExistingSkillsScanner (was matrix_composer_existing_skills_scanner.py)
      composer.py               ← SkillMatrixComposer (was matrix_composer.py)

    memory/                     ← Phase 1 (new)
      __init__.py
      scanner.py                ← MemorySectionScanner
      composer.py               ← MemoryMatrixComposer

    tools/                      ← Phase 2 (new)
      __init__.py
      scanner.py                ← ToolScanner
      composer.py               ← ToolMatrixComposer
```

Phase 1 adds `memory/`. Phase 2 adds `tools/`. Everything else is shared.

---

## What changes, what stays the same

### What moves (no logic change, just location)

| Old path | New path | Notes |
|---|---|---|
| `skill_matrix/fingerprint.py` | `recommendation_matrix/shared/fingerprint.py` | `SkillRecord` → `ComponentRecord` |
| `skill_matrix/state.py` | `recommendation_matrix/shared/state.py` | `SkillState` → `ComponentState` |
| `skill_matrix/matrix_composer_queries_generator.py` | `recommendation_matrix/shared/queries_generator.py` | Class name unchanged |
| `skill_matrix/matrix_composer_single_skill_evaluator.py` | `recommendation_matrix/shared/single_evaluator.py` | Class renamed to `SingleEvaluator` |
| `skill_matrix/matrix_composer_all_skills_evaluator.py` | `recommendation_matrix/shared/all_items_evaluator.py` | Class renamed to `AllItemsEvaluator` |
| `skill_matrix/matrix_composer_changed_skills_determiner.py` | `recommendation_matrix/shared/changed_items_determiner.py` | Class name unchanged |
| `skill_matrix/matrix_composer_state_saver.py` | `recommendation_matrix/shared/state_saver.py` | Updated to accept component type |
| `skill_matrix/matrix_composer_summary_printer.py` | `recommendation_matrix/shared/summary_printer.py` | Unchanged |
| `skill_matrix/metrics/` | `recommendation_matrix/shared/metrics/` | Entirely unchanged |
| `skill_matrix/matrix_composer_existing_skills_scanner.py` | `recommendation_matrix/skills/scanner.py` | Class name unchanged |
| `skill_matrix/matrix_composer.py` | `recommendation_matrix/skills/composer.py` | Class renamed to `SkillMatrixComposer` |
| `skill_matrix/cli.py` | `recommendation_matrix/cli.py` | Updated (see below) |
| `skill_matrix/__main__.py` | `recommendation_matrix/__main__.py` | Updated import path |

### What changes in logic

**`fingerprint.py`:** `SkillRecord` is generalized. The fields stay identical
(`name`, `description`, `body`, `mtime`, `directory`) — only the class is renamed to
`ComponentRecord` so it is not confusing when used for memory sections or tools.
`SkillRecord` can remain as a type alias (`SkillRecord = ComponentRecord`) for
backward compatibility within the skills module.

**`state.py`:** `SkillState` renamed to `ComponentState`. `MatrixState` gets an extra
`component_type: str` field so the state for skills, memory, and tools can be stored
in separate files without ambiguity.

**`single_evaluator.py`:** Class renamed from `SingleSkillEvaluator` to
`SingleEvaluator`. The logic is 100% identical — it only uses the component body as
the system message, which works for any component type.

**`all_items_evaluator.py`:** Class renamed from `AllSkillsEvaluator` to
`AllItemsEvaluator`. The `_write_matrix_file()` method gets a `component_type`
argument to prefix the output filename.

**`cli.py`:** Gets a `--type` argument:
```
python -m jiuwenswarm.tools.recommendation_matrix build --type skills
python -m jiuwenswarm.tools.recommendation_matrix build --type memory
python -m jiuwenswarm.tools.recommendation_matrix build --type tools
python -m jiuwenswarm.tools.recommendation_matrix build          ← builds all three
```

### What does NOT change

- The `oracle/` output directory (`~/.jiuwenswarm/agent/workspace/oracle`).
  All output files continue to go here.
- The JSON format inside each `scoring_matrix_*.json` file. Identical structure.
- All metric functions (f1, bigram_f1, bag_of_words, length_ratio).
- All fingerprinting logic.
- The `QueriesGenerator` logic (only the prompt template changes per component type).
- The `SingleEvaluator` logic (completely generic already).

---

## Output file naming

This is the one place where the rename does affect existing files.

**Current naming:** `scoring_matrix_<skill_name>.json`

**New naming:** `scoring_matrix_skill_<skill_name>.json`

A type prefix is added. This allows all three component types to share the same output
directory without the skill recommender accidentally treating memory matrix files as
skills.

| Component type | File pattern | Example |
|---|---|---|
| Skills | `scoring_matrix_skill_<name>.json` | `scoring_matrix_skill_email_sender.json` |
| Memory sections | `scoring_matrix_mem_<name>.json` | `scoring_matrix_mem_deployment_setup.json` |
| Tools | `scoring_matrix_tool_<name>.json` | `scoring_matrix_tool_web_search.json` |

**State files** also get a type suffix:
- `matrix_state_skills.json`
- `matrix_state_memory.json`
- `matrix_state_tools.json`

All in the same `oracle/` directory.

---

## One update needed in openjiuwen

The skill recommender (`RecommendSkillTool` in `openjiuwen`) currently reads all files
matching `scoring_matrix_*.json`. After the rename, it must read only
`scoring_matrix_skill_*.json` — otherwise it would try to use memory and tool matrix
files as skills.

This is a one-line change in the openjiuwen reader:

```python
# Before:
files = list(oracle_dir.glob("scoring_matrix_*.json"))

# After:
files = list(oracle_dir.glob("scoring_matrix_skill_*.json"))
```

This is the only change needed outside `jiuwenswarm/tools/`.

---

## Migration steps

1. Create `recommendation_matrix/` directory with the structure above
2. Move all files from `skill_matrix/` to their new locations
3. Rename classes (`SkillState` → `ComponentState`, etc.)
4. Add `component_type` prefix to output filenames in `AllItemsEvaluator`
5. Update `cli.py` with `--type` argument
6. Update the skill recommender glob in openjiuwen (one line)
7. Rename existing output files in `oracle/`:
   ```
   scoring_matrix_email_sender.json → scoring_matrix_skill_email_sender.json
   ```
   (Or just delete them and let the next build regenerate them.)
8. Delete `skill_matrix/` directory
9. Run: `python -m jiuwenswarm.tools.recommendation_matrix build --type skills`
   Verify output appears in `oracle/` with the new names.

---

## After Phase 0

The repository looks like this:

```
jiuwenswarm/tools/recommendation_matrix/
  shared/   ← all generic pipeline code
  skills/   ← skill-specific scanner + composer
  memory/   ← (empty, ready for Phase 1)
  tools/    ← (empty, ready for Phase 2)
```

Phase 1 fills in `memory/`. Phase 2 fills in `tools/`. Neither duplicates any code
from `shared/`.
