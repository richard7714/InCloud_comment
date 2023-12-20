#!/bin/bash
#SBATCH --time=01:59:00
#SBATCH --mem=128gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

# source <...>/miniconda3/etc/profile.d/conda.sh # Replace with path to your conda
# conda activate MinkLoc3D # Replace with name of your environment

_ROOT='/home/ma/git/incloud_comment' # Replace with root of your MinkLoc3D 
_YAML='4-step_helipr.yaml'
_GPU=0
########################################### Training Settings ################################################
_batch_size=256
# None, LwF, EWC, StructureAware
_incremental_name='EWC'
_SAVEDIR="${_ROOT}/results/Incloud_helipr_${_batch_size}_${_incremental_name}" # Replace with your save root 
##############################################################################################################

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=$PYTHONPATH:$_ROOT
cd $_ROOT

    # --incremental_environments 'pickles/Velodyne/Velodyne_train.pickle' 'pickles/Aeva/Aeva_train.pickle' 'pickles/Avia/Avia_train.pickle'\
# Replace --initial_environment and --incremental_environments args with your generated pickle files below 
CUDA_VISIBLE_DEVICES=$_GPU python3 training/train_incremental.py  \
    --initial_ckpt 'weights/minkloc3d_ouster_none.pth' \
    --initial_environment 'pickles/Ouster/Ouster_train.pickle'  \
    --incremental_environments  'pickles/Aeva/Aeva_train.pickle' 'pickles/Velodyne/Velodyne_train.pickle'\
    --config config/protocols/$_YAML \
    train.batch_size $_batch_size \
    train.memory.num_pairs 256 \
    train.loss.incremental.name $_incremental_name \
    train.loss.incremental.weight 110 \
    train.loss.incremental.adjust_weight True \
    save_dir $_SAVEDIR \
    # See config/default.yaml for a list of tunable hyperparameters!

