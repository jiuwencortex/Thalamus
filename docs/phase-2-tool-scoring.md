# Phase 2 — Tool Scoring

> **Prerequisite:** Phase 0 (rename and restructure) must be done first.
> Phase 1 (memory scoring) should also be done, but Phase 2 can technically run in
> parallel with Phase 1 — both fill slots in the same `recommendation_matrix/` package.
>
> Phase 2 fills in `recommendation_matrix/tools/` — two new files, using shared code.

---

## The problem with tools today

The agent sends the full list of available tools to the LLM on every call.
The tool list is filtered by a static `TOOL_WHITELIST` in
`jiuwenswarm/agents/swarm/providers/tools.py` and then sent as-is.

Current tools in the whitelist include:
- File operations: `read_file`, `write_file`, `edit_file`, `glob`, `grep`
- Execution: `bash_exec`
- Web: `web_search`, `web_fetch`, `web_paid_search`
- Skill management: `search_skill`, `install_skill`, `skill_branch_peek`, `skill_branch_explore`
- Symphony: `symphony_compose_score`, `symphony_refresh_score`
- Multimodal: `video_understanding`, `generate_image`
- Todos: `todo_create`, `todo_modify`, `todo_list`
- Device: `xiaoyi_phone` (when enabled)

For a query like "translate this paragraph to French", most of these are irrelevant.
Sending 20+ tool schemas adds noise and increases the chance the LLM tries to use a
tool it shouldn't, or gets confused about which one to reach for.

---

## Where the new code lives

After Phase 0, the directory structure is:

```
jiuwenswarm/tools/recommendation_matrix/
  shared/                    ← all generic pipeline code (from Phase 0)
  skills/                    ← skill-specific scanner + composer (from Phase 0)
  memory/                    ← memory-specific scanner + composer (from Phase 1)
  tools/                     ← Phase 2 adds these two files:
    __init__.py
    scanner.py               ← ToolScanner
    composer.py              ← ToolMatrixComposer
```

**That is all that Phase 2 adds.** Two new files in `tools/`. All scoring, matrix
writing, state management, and fingerprinting come from `shared/` unchanged.

---

## How tools differ from skills and memory

Skills have a SKILL.md body — a full page of instructions that is sent to the LLM as
a system message during evaluation. Memory sections have their section text.

Tools are different: they are defined by a JSON schema (name, description, parameters),
not by a body of text instructions. They are also exposed to the LLM through the
function-calling interface, not as text in the system prompt.

This creates a design choice in how to evaluate them.

### Evaluation approach: text proxy

For Phase 2, each tool's "body" is constructed as a plain-text summary:

```
Tool: web_search
Description: Search the web and return relevant results.
Parameters:
  - query (string, required): The search query to run
  - max_results (integer, optional): Maximum number of results (default 10)
```

This text becomes the `body` field of a `ComponentRecord` — the same field that holds
the SKILL.md content for skills and the section text for memory. The evaluation LLM
call is identical: system message = tool body, user message = generated query.

What this measures: **"if the LLM is told about this tool's capabilities, does it
produce a better answer for this type of query?"** This is a proxy for actual tool
usefulness. It is not as precise as running the tool for real, but it requires no
execution environment and reuses 100% of the shared pipeline.

A more precise evaluation (actually calling the tool and measuring result quality)
is described at the end as a Phase 2 upgrade path.

---

## `tools/scanner.py` — what it does

`ToolScanner` reads tool definitions from the agent's tool registry and returns one
`ComponentRecord` per tool (or tool group).

### Where tool definitions come from

Tool cards are registered in `jiuwenswarm/agents/swarm/providers/tools.py` and in the
openjiuwen core. Each tool has a `ToolCard` with: `name`, `description`, and a
parameters schema.

`ToolScanner` walks the same `TOOL_WHITELIST` and reads each tool card.

```python
ComponentRecord(
    name="web_search",
    description="Search the web and return relevant results.",
    body=(
        "Tool: web_search\n"
        "Description: Search the web and return relevant results.\n"
        "Parameters:\n"
        "  - query (string, required): The search query to run\n"
        "  - max_results (integer, optional): Maximum number of results (default 10)\n"
    ),
    mtime=<tool_registry_mtime>,
    directory=Path("jiuwenswarm/agents/swarm/providers"),
)
```

**Fingerprinting:** SHA256 of `(name + description + parameter_names_sorted)`.
A tool is considered changed if its name, description, or parameter list changes.
`mtime` is set to the mtime of the registry file that defines this tool.

### Tool groups

Some tools only make sense when used together. `skill_branch_peek` without
`skill_branch_explore` is incomplete. Sending one but not the other would break
the workflow.

The scanner supports a `TOOL_GROUPS` config:

```python
TOOL_GROUPS = [
    ["skill_branch_peek", "skill_branch_explore", "skill_index_build"],
    ["search_skill", "install_skill", "uninstall_skill"],
    ["symphony_compose_score", "symphony_refresh_score"],
]
```

When tools are grouped, the scanner creates **one `ComponentRecord` per group** whose
`name` is the group name (e.g. `"skill_browsing"`) and whose `body` combines all
tool descriptions in the group. At query time, if the group is selected, all tools
in the group are included.

---

## `tools/composer.py` — what it does

`ToolMatrixComposer` is the orchestrator for tool scoring. Structurally identical to
`SkillMatrixComposer` and `MemoryMatrixComposer`:

```python
class ToolMatrixComposer:
    def __init__(self, matrix_dir, model, model_name, n_examples, parallel):
        self._scanner    = ToolScanner()
        self._determiner = ChangedItemsDeterminer(matrix_dir,
                                                  state_file="matrix_state_tools.json")
        self._generator  = QueriesGenerator(model, model_name, n_examples, parallel,
                                            prompt_template=_TOOL_PROMPT_TEMPLATE)
        self._evaluator  = AllItemsEvaluator(model, model_name, matrix_dir, parallel,
                                             file_prefix="tool")
        self._saver      = StateSaver(matrix_dir, state_file="matrix_state_tools.json")

    async def build(self, force=False, only=None):
        tools = self._scanner.scan(only)
        changed, skipped = self._determiner.determine(tools, force)
        gen_results = await self._generator.generate_for_items(changed)
        states, llm_calls = await self._evaluator.evaluate_all(gen_results)
        self._saver.save(tools, skipped, states)
```

Only two things differ from `MemoryMatrixComposer`:
1. The scanner (`ToolScanner`)
2. The query generation prompt template

---

## The query generation prompt

```
You are given an AI agent tool — its name, description, and parameters.
Generate {n} realistic user requests where this tool would be useful.

Each example must have:
  - "query": a natural user message (1-3 sentences) where using this tool
    would help produce a good response
  - "answer": what a good response looks like when the tool is available
    (1-2 sentences)

Tool name: {name}
Description: {description}

Tool details:
{body}

Requirements:
- Queries must be realistic user requests, not descriptions of the tool itself.
- Cover different use cases and parameter combinations of this tool.
- Do not generate queries that are outside the scope of what this tool does.

Return a JSON array of exactly {n} objects:
[{{"query": "...", "answer": "..."}}, ...]
No other text.
```

Uses the same `{name}`, `{description}`, `{body}` placeholders as skill and memory
templates. `QueriesGenerator` in `shared/` accepts any template — no changes needed.

---

## Output files

Tool matrix files go into the **same `oracle/` directory** as skill and memory files.
Distinguished by a `tool_` prefix.

```
~/.jiuwenswarm/agent/workspace/oracle/
  scoring_matrix_skill_email_sender.json        ← skills (Phase 0)
  scoring_matrix_mem_architecture_overview.json ← memory (Phase 1)
  scoring_matrix_tool_web_search.json           ← tools (Phase 2)
  scoring_matrix_tool_bash_exec.json
  scoring_matrix_tool_skill_browsing.json       ← tool group
  scoring_matrix_tool_symphony.json             ← tool group
  matrix_state_skills.json
  matrix_state_memory.json
  matrix_state_tools.json                       ← tools state (Phase 2)
```

File format is identical to skills and memory. Added fields:

```json
{
  "run_id": "tool_web_search_20251014T090000",
  "component_type": "tool",
  "component_name": "web_search",
  "is_group": false,
  "group_members": [],
  "fitness_metrics": ["f1", "bigram_f1", "bag_of_words", "length_ratio"],
  "baseline_cross_eval": [
    {
      "example_id": "tool_web_search_0000",
      "example_input": "Find the latest Python release notes and summarize them",
      "example_expected": "Here are the main changes in the latest Python release...",
      "candidate_output": "Based on my search capability, the latest Python release...",
      "scores": { "f1": 0.63, "bigram_f1": 0.58, "bag_of_words": 0.61, "length_ratio": 0.87 }
    }
  ],
  "evolved_cross_eval": []
}
```

`is_group` and `group_members` are new fields. For non-grouped tools both are empty.
For a group like `skill_browsing`, `group_members = ["skill_branch_peek", "skill_branch_explore"]`.

---

## CLI

```bash
# Build only tool matrix
python -m jiuwenswarm.tools.recommendation_matrix build --type tools \
    --matrix-dir ~/.jiuwenswarm/agent/workspace/oracle \
    --model gpt-4o-mini \
    --api-key $OPENAI_API_KEY \
    --n-examples 15

# Build everything (skills + memory + tools) in one command
python -m jiuwenswarm.tools.recommendation_matrix build \
    --project-dir ~/.jiuwenswarm \
    --skills-dir ~/.jiuwenswarm/skills \
    --matrix-dir ~/.jiuwenswarm/agent/workspace/oracle \
    --model gpt-4o-mini \
    --api-key $OPENAI_API_KEY
```

---

## How selection works at query time

A `ToolRecommender` reads `scoring_matrix_tool_*.json` and selects which tools to
expose per query. Same TF-IDF cosine similarity approach as skills and memory.

**Safety rule:** If no tool scores above the similarity threshold, or if the matrix
is missing, fall back to the full whitelist. The agent never loses all its tools due
to a bad matrix.

**Group expansion:** When a tool group is selected, all its member tools are included
in the function calling interface — never just some of them.

**Integration point:** `jiuwenswarm/agents/swarm/providers/tools.py`.
The provider currently returns the full whitelist. With Phase 2, it calls the
`ToolRecommender` and returns only the selected subset.

---

## Complete oracle directory after Phases 0–2

```
~/.jiuwenswarm/agent/workspace/oracle/
  scoring_matrix_skill_*.json      ← one per installed skill
  scoring_matrix_mem_*.json        ← one per memory section
  scoring_matrix_tool_*.json       ← one per tool / tool group
  matrix_state_skills.json
  matrix_state_memory.json
  matrix_state_tools.json
```

Three families of files, all in one directory, distinguishable by filename prefix.
The existing skill recommender reads only `scoring_matrix_skill_*.json`.
New memory and tool recommenders read their respective prefixes.

---

## LLM cost estimate

With ~12 tools/groups and `n_examples=15`:

| Step | Calls |
|---|---|
| Query generation (Stage 1) | 12 × 1 = **12 calls** |
| Evaluation (Stage 2) | 12 × 15 = **180 calls** |
| Total | **192 LLM calls** |

Tools change rarely (only when developers add new tools). Rebuilds are infrequent.

---

## Implementation checklist

| Task | File | Shared code reused |
|---|---|---|
| Tool scanner | `tools/scanner.py` | Uses `ComponentRecord` from `shared/fingerprint.py` |
| Tool group config | `tools/scanner.py` | `TOOL_GROUPS` list |
| Query prompt template | `tools/composer.py` | Passed to `QueriesGenerator` in `shared/` |
| Composer orchestrator | `tools/composer.py` | Uses all of `shared/` |
| `--type tools` in CLI | `cli.py` | Routes to `ToolMatrixComposer` |
| Output file naming | `shared/all_items_evaluator.py` | `file_prefix="tool"` parameter |
| State file | `shared/state_saver.py` | `state_file="matrix_state_tools.json"` |
| `ToolRecommender` | new file in `recommendation_matrix/` | Reads `scoring_matrix_tool_*.json` |
| `tools.py` provider update | `agents/swarm/providers/tools.py` | Calls `ToolRecommender` |
| Group expansion | `ToolRecommender` | Expands group_members at selection time |

**Total new code in `tools/`: ~150 lines.**
Everything else is parameter changes to existing shared code.

---

## Upgrade path: actual tool execution evaluation

The text-proxy evaluation (Phase 2 as described above) measures whether knowing about
a tool helps produce better answers. A more accurate alternative runs the tool for real:

1. Generate queries where the tool would be called
2. Run LLM with full tool schema in function-calling interface
3. LLM either calls the tool or it doesn't (measure: invocation rate)
4. If called, execute the tool and check the result (measure: parameter correctness)
5. Compare final response with and without tool available (measure: response quality delta)

This gives three signals instead of one. It requires a real execution environment and
is significantly more complex. The right time to implement this upgrade is after the
text-proxy version is running and the tool list has grown to 30+ tools where the noise
of irrelevant tools becomes measurable.

---

## What Phase 2 unlocks

- LLM sees only relevant tools → fewer wrong tool calls, less confusion
- Smaller function-calling schema → token savings per request
- Tool group coherence: related tools are always selected or skipped together
- Foundation for Phase 3: tool scores become inputs to the evolutionary search
- Over time, the tool matrix reveals which tools are actually useful for which tasks
  — valuable for prioritizing future tool development
