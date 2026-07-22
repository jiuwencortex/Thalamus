# Enables:
#   python -m context_selectors lookup ...
#   python -m context_selectors classify ...
#
# Examples:
#   python -m context_selectors lookup \
#       --oracle-dir ~/.jiuwenswarm/oracle
#
#   python -m context_selectors lookup \
#       --oracle-dir ~/.jiuwenswarm/oracle \
#       --query "Set up a CI pipeline for my new service"
#
#   python -m context_selectors classify \
#       --oracle-dir ~/.jiuwenswarm/oracle
#
#   python -m context_selectors classify \
#       --oracle-dir ~/.jiuwenswarm/oracle \
#       --embedding ./query.npy --threshold 0.5 --verbose

from context_selectors.cli import main

main()
