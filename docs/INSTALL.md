# Installation


#### Requirements
- Python ≥ 3.7
- PyTorch ≥ 1.8
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

The code is tested with PyTorch 1.10.0 and CUDA 11.3. After cloning the repository, 
follow the below steps for installation:

1. Create virtual environment:
````shell
# conda
       
conda update conda
conda create -n detector python=3.7.7
conda activate detector
        
# or virtualenv
       
sudo apt install python3.7 python3-venv python3.7-venv
python3.7 -m venv detector
. detector/bin/activate
 pip install --upgrade pip
````

2. Install torch:
```shell
# conda: 
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
    
# or pip:
pip install torch==1.10.0 torchvision==0.11.0 torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

3. Install Detectron2
Install with [pre-build Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only):
```shell
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```
 or install from source:
```shell
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..
```
4. Install other dependencies
```shell
cd dph-object-detector
pip install -r requirements.txt
```
