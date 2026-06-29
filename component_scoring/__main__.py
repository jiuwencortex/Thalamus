# Enables: python -m jiuwenswarm.tools.component_scoring [build] --type skills|memory|tools
#
# Examples:
#   python -m jiuwenswarm.tools.component_scoring build --type skills \
#       --skills-dir ~/.jiuwenswarm/agent/workspace/skills \
#       --matrix-dir ~/.jiuwenswarm/agent/workspace/oracle \
#       --model gpt-4o-mini --api-key $OPENAI_API_KEY
#
#   python -m jiuwenswarm.tools.component_scoring build \
#       --skills-dir ~/.jiuwenswarm/agent/workspace/skills \
#       --matrix-dir ~/.jiuwenswarm/agent/workspace/oracle \
#       --model gpt-4o-mini --api-key $OPENAI_API_KEY

# # Dry-run (no API key needed):
# python -m jiuwenswarm.tools.component_scoring build --type skills \
#     --skills-dir ~/.jiuwenswarm/skills --matrix-dir ~/.jiuwenswarm/oracle \
#     --model gpt-4o-mini --dry-run
#
# # Build all three:
# python -m jiuwenswarm.tools.component_scoring build --skills-dir ~/.jiuwenswarm/skills --project-dir ~/.jiuwenswarm --tools-config ~/.jiuwenswarm/tool_definitions.yaml --matrix-dir ~/.jiuwenswarm/oracle --model gpt-4o-mini --api-key $KEY
# python -m jiuwenswarm.tools.component_scoring build --skills-dir C:\Users\m00645993\.jiuwenswarm\agent\workspace\skills --project-dir C:\Users\m00645993\.jiuwenswarm\agent\workspace --tools-config ~/.jiuwenswarm/tool_definitions.yaml --matrix-dir C:\Users\m00645993\.jiuwenswarm\agent\workspace\oracle --model "kimi-k2.6" --api-key "sk-s89oB4Ni1ZQjnDyN5_ZDkA" --api-base "https://litellm.toga-ai.toganetworks.com/v1" --n-examples 20

from component_scoring.cli import main

main()
