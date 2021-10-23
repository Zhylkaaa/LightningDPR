conda create -n deepspeed_env python=3.8
conda activate deepspeed_env
#conda install pytorch cudatoolkit=11.1 cudatoolkit-dev=11.1 compilers -c pytorch -c nvidia -c conda-forge
conda install pytorch cudatoolkit=11.1 cudatoolkit-dev=11.1 gcc gxx -c pytorch -c nvidia -c conda-forge

#python -m pip install transformers pytorch_lightning sentencepiece deepspeed wandb fairscale
python -m pip install sentencepiece deepspeed wandb fairscale