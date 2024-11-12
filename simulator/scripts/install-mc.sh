#!/bin/bash

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root"
    exit
fi

if ! command -v mc &> /dev/null
then
    pushd /usr/local/bin
    wget https://dl.min.io/client/mc/release/linux-amd64/mc
    chmod +x mc
    popd
else
    mc --version
fi
