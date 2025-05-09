#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --nodes=1                # Request 1 node
#SBATCH --ntasks=1               # Request 1 task
#SBATCH --mem=2G                 # Request 2 GB of memory
#SBATCH --job-name=test      # Specify the job name
#SBATCH --output=output.txt      # Specify the output file
#SBATCH --error=error.txt        # Specify the error file
#SBATCH --mail-user=carrlosaperrez@gmail.com # Email address for job notifications
#SBATCH --mail-type=ALL          # Receive email notifications for all job events



# Load virtualenv
module load gcc/9.4.0 python/3.13.2 py-virtualenv/16.7.6 scipy-stack
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt

python print("Hello World")