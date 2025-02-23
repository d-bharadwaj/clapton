#!/bin/bash
## sample script to run single node cpu task
#SBATCH -A m4669
## specify cpu or gpu node at next line
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 10:00:00
## node numbers
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
## change the job name to your script name
## SBATCH --output=../logs/%x_%j.log
## change the username to your username
#SBATCH --mail-user=dhanvib

## load required modules, if not found, install them perferably under /global/common/software/m4669
## nproc
module load conda
module load python
## assume you have a virtual environment in the current directory named env

# Check if a file was passed as an argument
if [ -z "$1" ]; then
    echo "No Python script specified. Exiting."
    exit 1
fi

# Activate conda venv. 
conda activate pauli_twirl

# Run the Python script passed as the argument
# python "$1".py $N_QUBITS $N_REPS
srun --cpu-bind=cores python "$1".py 

## srun python3 runtime_profile.py
echo 'job finished!'

## just add srun to any compute intensive task to run it on nodes

## run 'squ' to monitor the pending/running job
## run 'sacct' to see all jobs status
## run 'scancel <jobid>' to cancel a job