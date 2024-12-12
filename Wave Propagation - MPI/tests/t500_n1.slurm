#!/bin/bash
#SBATCH --job-name="../wave260"
#SBATCH --output="t500_n1.%j.%N.out"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0G
#SBATCH --account=csd911
# #SBATCH --export=None
#SBATCH --export=ALL
#SBATCH -t 0:03:00
####   #SBATCH --mail-type=BEGIN,END,FAIL
####   #SBATCH --mail-user=your_email@ucsd.edu

# setup your environment

export SLURM_EXPORT_ENV=ALL
export MV2_USE_RDMA_CM=0
export MV2_HOMOGENEOUS_CLUSTER=1

module purge
module load cpu/0.17.3b  gcc/10.2.0/npcyll4  mvapich2/2.3.7/iyjtn3x
module load cmake/3.21.4/n5jtjsf
module load netcdf-c/4.8.1/yt65vte
module load ncview/2.1.8/znk6ds3
module load tau/2.30.2/auogdal
module load slurm

mkdir -p /expanse/lustre/scratch/azhu5/temp_project/job_$SLURM_JOB_ID

srun --chdir /expanse/lustre/scratch/azhu5/temp_project/job_$SLURM_JOB_ID --mpi=pmi2 -n 1 $SLURM_SUBMIT_DIR/../build_expanse/wave260mpi $SLURM_SUBMIT_DIR/../tests/t500.config  -y 1 -x 1  -i 10000  
    