#!/bin/bash
#SBATCH -J sanity
#SBATCH -o sanity_%j.out
#SBATCH -e sanity_%j.err
#SBATCH --mail-user=brytech@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --exclude=node0054
## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=nlp
CONDA_ROOT=$HOME2/anaconda3
## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited
## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST
## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date
## Horovod execution
## chmod +x job1.sh
## ./job1.sh
CMD="sleep 12h"
echo $CMD
$CMD
echo "Run completed at:- "
date