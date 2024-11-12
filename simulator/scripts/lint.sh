#!/bin/bash
# Install: pip install cpplint

PROJECT_DIR="$( cd "$( dirname "$0" )/.." && pwd -P )"

pushd $PROJECT_DIR
cpplint \
  --filter=\
-legal/copyright,\
-build/header_guard,\
-build/include_order,\
-runtime/references,\
-runtime/explicit,\
-whitespace/end_of_line,\
-whitespace/parens \
  --exclude=./include/protobuf/aixh.pb.h \
  --exclude=./include/aixh.pb.h \
  --exclude=./include/spear.proto.e8e8.pb.h \
  --exclude=./include/cudnn/common/gsl-lite.hpp \
  --exclude=./include/stb_image.h \
  --exclude=./include/stb_image_write.h \
  --recursive \
  ./include/ > scripts/lint_result.include.txt 2>&1

cpplint \
  --filter=\
-legal/copyright,\
-build/header_guard,\
-build/include_order,\
-runtime/references,\
-runtime/explicit,\
-whitespace/end_of_line,\
-whitespace/parens,\
-readability/casting \
  --exclude=./src/core/aixh.pb.cc \
  --exclude=./src/core/spear.proto.e8e8.pb.cc \
  --recursive \
  ./src/ > scripts/lint_result.src.txt 2>&1

popd
