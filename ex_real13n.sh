#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 16 processor core(s) per node
#SBATCH --job-name="128-13"
#SBATCH --output="s%j.out" # job standard output file (%j replaced by job id)
#
. ../../cnn/bin/activate
cd /home/aapowadi/anirudha/m_exps/over_fit_128/

python ex_real.py -i ex13n
python ex_real.py -i ex13n_valid