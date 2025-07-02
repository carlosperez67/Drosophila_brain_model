#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=def-mdgordon-ab
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=grn_sweeps
#SBATCH --output=/scratch/cperez67/logs/grn_sweeps_%A_%a.out
#SBATCH --error=/scratch/cperez67/logs/grn_sweeps_%A_%a.err
#SBATCH --mail-user=carrlosaperrez@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --array=35-50:5           # sugar‑Hz = 35,40,45,50

module load python/3.10.13 scipy-stack/2025a
source ENV/bin/activate

# map the array index to sugar frequency
SUGAR_HZ=${SLURM_ARRAY_TASK_ID}

# build the ORN list: 20,30,…,200
ORN_HZ_LIST=$(seq 20 10 200)

# results directory
RES_DIR="/scratch/cperez67/grn_sweeps/grn_sweeps_${SUGAR_HZ}Hz"
mkdir -p "${RES_DIR}"

cd $SLURM_SUBMIT_DIR/Drosophila_brain_model

python sweetActivation.py \
  --subset both \
  --sugar-hz "${SUGAR_HZ}" \
  --orn-hz ${ORN_HZ_LIST} \
  --res-dir "${RES_DIR}"

deactivate