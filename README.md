# active_learning_segmentation

### How to run:
Go to trainer folder and run: 

$ python main.py -acq 0 -cuda 0 

where: 

acq value:

        0: "cfe",
        
        1: "mfe",
        
        2: "std",
        
        3: "mi",
        
        4: "random",

### Build docker to run:
#### cd vao trong thu muc chua Dockerfile :
$ docker build -t activelr .
$ docker run -ti -v $PWD:/code activelr:latest /bin/bash
#### ben trong docker
$ source ~/miniconda3/etc/profile.d/conda.sh
$ conda activate pytorch
$ cd code/act_lear_2/trainer
$ python main.py -acq 1
