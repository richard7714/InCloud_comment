#!/bin/bash
#SBATCH --time=01:59:00
#SBATCH --mem=128gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1

# source <...>/miniconda3/etc/profile.d/conda.sh # Replace with path to your conda
# conda activate MinkLoc3D # Replace with name of your environment

# _ROOT='/home/ma/git/incloud_comment' # Replace with root of your MinkLoc3D 
_ROOT='//home/ma/git/incloud_comment' # Replace with root of your MinkLoc3D 
_SAVEDIR="${_ROOT}/results/InCloud_MinkLoc3D_Oxford_inhouse" # Replace with your save root 

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=$PYTHONPATH:$_ROOT
cd $_ROOT

# Replace --initial_environment and --incremental_environments args with your generated pickle files below 
python3 training/train_incremental.py  \
    --initial_ckpt 'weights/minkloc3d_baseline.pth' \
    --initial_environment 'pickles/Oxford/Oxford_train_queries.pickle' \
    --incremental_environments 'pickles/In-house/In-house_train_queries.pickle' 'pickles/In-house/In-house_train_queries.pickle' \
    --config config/protocols/4-step.yaml \
    train.memory.num_pairs 256 \
    train.loss.incremental.name 'StructureAware' \
    train.loss.incremental.weight 110 \
    train.loss.incremental.adjust_weight True \
    save_dir $_SAVEDIR \
    # See config/default.yaml for a list of tunable hyperparameters!
    



