#!/bin/bash

PROJECT_DIR="$( cd "$( dirname "$0" )/.." && pwd -P )"

pushd ${PROJECT_DIR}/dump

for file in *;
do
  cmp $file ../dump_darknet/$file
  if [ $? -eq 0 ]; then
    echo "OK: $file"
  fi
done

popd
