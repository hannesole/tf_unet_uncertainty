#!/bin/bash
#PBS -N std_py
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:nvidiaTITANX,walltime=24:00:00,mem=12gb
#PBS -q default-gpu
#PBS -m abe
#PBS -M hh128@jupiter.uni-freiburg.de
#PBS -j oe
#might want to change to nodes=1:nvidiaTITAN:ppn=1,gpus=1,walltime=23:59:59 to train on Titan only
#or use hostlist=^chip+chap+dicky+ducky+rattcapone+summi+track+trixi+william,nodes=1:nvidiaMin12GB:ppn=12,gpus=1,walltime=23:59:59

# usage in commandline: bashscript.sh mode
export SCRIPT_NAME=std_unet.py
export MODE=train
export NAME=${name}
export OUT_DIR="/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet/output_scr"
export TRAIN_DIR=$OUT_DIR'/'$NAME
echo $MODE $TRAIN_DIR

#source /misc/software/cuda/add_environment_cuda9.0.176_cudnnv7.sh      # used for Tensorflow >= 1.5.0
source /misc/software/cuda/add_environment_cuda8.0.61_cudnnv6.sh        # used for Tensorflow < 1.5.0

export PYTHONUNBUFFERED=1
export PATH=/misc/lmbraid12/hornebeh/std/py_env/tf_main/bin:$PATH
export PYTHONPATH=/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet:/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet/helper:/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet/unet:/misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet/util

# change to project dir
cd /misc/lmbraid19/hornebeh/std/projects/remote_deployment/win_tf_unet

# set python interpreter to virtual_evnironment
#. /misc/lmbraid19/hornebeh/std/py_env/tf_unet/bin/activate
source /misc/lmbraid19/hornebeh/std/py_env/tf_unet/bin/activate
python $SCRIPT_NAME -t $TRAIN_DIR -m $MODE 2>&1| tee $OUT_DIR/qlog-`date +%F_%R`.log