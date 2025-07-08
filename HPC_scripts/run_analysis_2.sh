#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --account=def-mdgordon-ab
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=Drosophila
#SBATCH --output=/scratch/cperez67/logs/Analysis/Drosophila_%A_%a.out
#SBATCH --error=/scratch/cperez67/logs/Analysis/Drosophila_%A_%a.err
#SBATCH --mail-user=carrlosaperrez@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=5-20:5           # sugar‑Hz = 35,40,45,50
#SBATCH --signal=B:USR1@500       # SIGUSR1 120s before timeout

module load python/3.10.13 scipy-stack/2025a
source ENV/bin/activate

DEFAULT_BASE="/project/def-mdgordon-ab/cperez67/low_grn_sweeps"
BASE="${1:-$DEFAULT_BASE}"

# optional ORN types after $1
if [ "$#" -gt 1 ]; then
  ORN_TYPES=( "${@:2}" )
  ORN_ARG=(--orn-types "${ORN_TYPES[@]}")
else
  ORN_ARG=()
fi

SUGAR_HZ=${SLURM_ARRAY_TASK_ID}
RES_DIR="${BASE}/grn_sweeps_${SUGAR_HZ}Hz"
TMP_RES="${SLURM_TMPDIR}/grn_sweeps_${SUGAR_HZ}Hz"

# make sure both exist
mkdir -p "${RES_DIR}" "${TMP_RES}"

# sync in only that sugar‑Hz folder
if [ -d "${RES_DIR}" ]; then
  echo "[$(date)] Sync in: ${RES_DIR} → ${TMP_RES}"
  cp -r "${RES_DIR}/." "${TMP_RES}/"
fi

# cleanup: always push TMP_RES back to RES_DIR
cleanup(){
  echo "[$(date)] Sync out: ${TMP_RES} → ${RES_DIR}"
  cp -r "${TMP_RES}/." "${RES_DIR}/"
}
trap 'echo "[$(date)] SIGUSR1"; cleanup; exit 0' USR1
trap 'echo "[$(date)] EXIT"; cleanup' EXIT

echo "[$(date)] Running sugar=${SUGAR_HZ}Hz, ORN_TYPES=${ORN_TYPES[*]:-all}"
cd "${SLURM_SUBMIT_DIR}/Drosophila_brain_model"

python sweetActivation.py \
  --subset both \
  --sugar-hz "${SUGAR_HZ}" \
  --orn-hz $(seq 30 10 40) \
  --res-dir "${TMP_RES}" \
  --reverse
  "${ORN_ARG[@]}"

deactivate