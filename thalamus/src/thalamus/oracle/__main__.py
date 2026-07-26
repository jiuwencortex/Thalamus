# Enables: python -m thalamus.oracle build
#
# Examples:
#   python -m thalamus.oracle build \
#       --oracle-dir ~/.jiuwenswarm/oracle
#
#   python -m thalamus.oracle build \
#       --oracle-dir ~/.jiuwenswarm/oracle \
#       --n-clusters 15 --population 100 --generations 200
#
#   python -m thalamus.oracle inspect \
#       --oracle-dir ~/.jiuwenswarm/oracle \
#       --query "Set up a CI pipeline for my new service"

#python -m thalamus.scoring build --type tools --tools-dir C:\Workspace\openjiuwen\agent-core\openjiuwen\harness\tools --tools-dir C:\Workspace\openjiuwen\jiuwenswarm\jiuwenswarm\agents\harness\common\tools  --matrix-dir C:\Users\m00645993\.jiuwenswarm\agent\workspace\oracle --model gpt-4o-mini --api-key $KEY

from .cli import main

main()
