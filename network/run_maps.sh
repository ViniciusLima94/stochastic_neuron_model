#!/bin/bash

#SBATCH -J MAPS_NET                      # Job name
#SBATCH -o out/MAPS NET.out              # Name of stdout output file (%j expands to %jobID)
#SBATCH -N 1                             # Total number of nodes requested
#SBATCH -n 40                           # Total number of mpi tasks requested
#SBATCH -t 700:00:00                     # Run time (hh:mm:ss) - 1.5 hours

python3 compute_maps.py
