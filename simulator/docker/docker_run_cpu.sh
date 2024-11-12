#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

docker run -it \
  --name spgraph-simulator \
  -v $SCRIPT_DIR/..:/root/simulator simulator-cpu /bin/bash
