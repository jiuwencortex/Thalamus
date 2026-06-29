# Enables:
#   python -m jiuwenswarm.thalamus.context_selectors lookup ...
#   python -m jiuwenswarm.thalamus.context_selectors classify ...
#
# Examples:
#   python -m jiuwenswarm.thalamus.context_selectors lookup \
#       --oracle-dir ~/.jiuwenswarm/oracle
#
#   python -m jiuwenswarm.thalamus.context_selectors lookup \
#       --oracle-dir ~/.jiuwenswarm/oracle \
#       --query "Set up a CI pipeline for my new service"
#
#   python -m jiuwenswarm.thalamus.context_selectors classify \
#       --oracle-dir ~/.jiuwenswarm/oracle
#
#   python -m jiuwenswarm.thalamus.context_selectors classify \
#       --oracle-dir ~/.jiuwenswarm/oracle \
#       --embedding ./query.npy --threshold 0.5 --verbose

from context_selectors.cli import main

main()
