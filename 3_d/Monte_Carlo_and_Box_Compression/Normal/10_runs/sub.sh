#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --partition=serial
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


# Environment setup
module purge
module load gcc/7.3.1

#python simple_MC_extended_object_NORMAL_compress.py
#python from_x_y_theta_to_vmd_readable.py
python extend_simple_MC_extended_object_NORMAL_compress.py 
