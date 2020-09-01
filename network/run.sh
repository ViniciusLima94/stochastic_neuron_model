#!/bin/bash

#SBATCH -J REDE_GL                      # Job name
#SBATCH -o out/REDE_GL.out              # Name of stdout output file (%j expands to %jobID)
#SBATCH -N 1                            # Total number of nodes requested
#SBATCH -n 1                           # Total number of mpi tasks requested
#SBATCH -t 700:00:00                    # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --array=0-4

python3 network.py $SLURM_ARRAY_TASK_ID
