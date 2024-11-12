#!/bin/bash

PROJECT_DIR="$( cd "$( dirname "$0" )/.." && pwd -P )"

pushd $PROJECT_DIR
if ! command -v mc &> /dev/null
then
    echo "Command 'mc' not found."
    echo "Please run command below."
    echo ""
    echo "sudo ${$PROJECT_DIR}/scripts/install-mc.sh"
    echo ""
    exit
else
    mc alias set yj \
        https://dudaji-disk.synology.me:49000 \
        $MINIO_ACCESS_KEY \
        $MINIO_SECRET_KEY
fi

# Download models 
mc cp yj/aichip/validation_models.tar.gz .
tar -zxvf validation_models.tar.gz
rm validation_models.tar.gz
mkdir -p models
mv validation_models models

# Download ImageNet Dataset
mkdir -p images
cd images
mc cp yj/aichip/imagenet.tar.gz .
tar -zxvf imagenet.tar.gz
rm -rf imagenet.tar.gz
mc cp yj/aichip/imagenet_validation.tar.gz .
tar -zxvf imagenet_validation.tar.gz
rm -rf imagenet_validation.tar.gz
mc cp yj/aichip/images5.tar.gz .
tar -zxvf images5.tar.gz
rm -rf images5.tar.gz
mkdir -p image1
cp images5/dog.jpg image1/dog.jpg
cd ..

# Set Other Directories
mkdir -p results
mkdir -p dump

popd
