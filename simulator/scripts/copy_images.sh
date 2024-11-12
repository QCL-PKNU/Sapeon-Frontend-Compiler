#!/bin/bash

PROJECT_DIR="$( cd "$( dirname "$0" )/.." && pwd -P )"

pushd $PROJECT_DIR

source_directory=${PROJECT_DIR}/images/imagenet
target_directory=${PROJECT_DIR}/images/test_images
number_of_files_to_copy=$1

# 목적지 디렉토리가 없으면 생성
rm -rf "${target_directory}" 
mkdir -p "${target_directory}"

# source_directory에서 number_of_files_to_copy 만큼의 파일을 찾아서 복사
count=0
for file in "$source_directory"/*; do
    if [ -f "$file" ]; then
        cp "$file" "$target_directory"
        ((count++))
    fi

    # number_of_files_to_copy에 도달하면 반복 중지
    if [ "$count" -eq "$number_of_files_to_copy" ]; then
        break
    fi
done

popd
