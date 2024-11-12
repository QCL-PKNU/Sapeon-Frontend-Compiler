local_dir=/home

docker run -it --name aix --gpus all -v $local_dir:/home nvidia/cuda:11.4.0-aix /bin/bash
