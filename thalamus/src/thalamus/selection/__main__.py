# Enables:
#   python -m thalamus.selection lookup ...
#   python -m thalamus.selection classify ...
#
# Examples:
#   python -m thalamus.selection lookup \
#       --oracle-dir ~/.jiuwenswarm/oracle
#
#   python -m thalamus.selection lookup \
#       --oracle-dir ~/.jiuwenswarm/oracle \
#       --query "Set up a CI pipeline for my new service"
#
#   python -m thalamus.selection classify \
#       --oracle-dir ~/.jiuwenswarm/oracle
#
#   python -m thalamus.selection classify \
#       --oracle-dir ~/.jiuwenswarm/oracle \
#       --embedding ./query.npy --threshold 0.5 --verbose

from .cli import main

main()
