#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # 16 processor core(s) per node
#SBATCH --job-name="real128"
#SBATCH --output="s%j.out" # job standard output file (%j replaced by job id)
#
. ../../cnn/bin/activate
cd /home/aapowadi/anirudha/m_exps/over_fit_128/
#python ex_real.py -i ex1n
#
#python ex_real.py -i ex1n_valid
#
#python ex_real.py -i ex2n
#
#python ex_real.py -i ex2n_valid
#
#python ex_real.py -i ex3n
#
#python ex_real.py -i ex3n_valid
#
python ex_real.py -i ex4n

python ex_real.py -i ex4n_valid

python ex_real.py -i ex5n

python ex_real.py -i ex5n_valid
#
#python ex_real.py -i ex6n
#
#python ex_real.py -i ex6n_valid
#
#python ex_real.py -i ex7n
#
#python ex_real.py -i ex7n_valid