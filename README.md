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



### Các tham số ảnh hưởng đến hiệu năng của các hàm acquisition (in my opinion):
- lúc tranining model qua các phase, model càng to thì nên càng đê nhiều epoches lúc training. 

- lúc eval thì càng để nhiều gauss_iter càng tốt.

- Nhược điểm là để các gt đó càng lớn thì càng lâu.

- Ngoài việc tính acquisition như trên, có thể combine các hàm lại với các trọn số khác nhau (cũng là 1 hyper param) để ra 1 hàm mới.

- Ngoài việc tính uncertainty của model, để collect data cho phase tiêp theo, ta có thể làm thêm 1 hàm để tính similarity giữa các ảnh, loại bỏ những unlabeled images mà similar với images trong labeled pool data.


