#!/bin/bash --login
#SBATCH --job-name=capstone
#SBATCH --output=test_logs/%j_capstone.out
#SBATCH --error=test_logs/%j_capstone.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

###############################################################################
# Environment Setup
###############################################################################

cd /data/shelbyho/capstone

# Exit on error for setup and verification
set -e

# Manually sync home directory and data directory
cp -r ~/capstone/model/ ./
cp ~/capstone/uv.lock ./
cp ~/capstone/pyproject.toml ./

# Environment config
export HF_TOKEN=hf_...
export HF_HOME=/data/shelbyho/.cache/huggingface
export UV_CACHE_DIR=/data/shelbyho/.cache/uv
uv sync
source .venv/bin/activate

# Parse command-line arguments
L1="$1"
shift
L2="$1"
shift

# Optional arguments
while [[ $# -gt 0 ]]; do
	case "$1" in 
		--size)
			SIZE="$2"
			shift 2
			;;
		--model)
			MODEL="$2"
			shift 2
			;;
		--finetune)
			FINETUNE="$2"
			shift 2
			;;
		--split)
			SPLIT="$2"
			shift 2
			;;
		--quantize)
			QUANTIZE="$2"
			shift 2
			;;
		*)
			echo "Unknown option: $1"
			exit 1
			;;
	esac
done

# Set default arguments
if [ -z "$SIZE" ]; then
	SIZE="base"
fi
if [ -z "$MODEL" ]; then
	MODEL="t5"
fi
if [ -z "$FINETUNE" ]; then
	FINETUNE="0"
fi
if [ -z "$SPLIT" ]; then
	SPLIT="10"
fi 
if [ -z "$QUANTIZE" ]; then
	QUANTIZE="4bit"
fi 

mkdir -p test_logs

# Make output directory
DIR_DATE=$(date +'%y%m%d')
OUTPUT_ROOT=/data/shelbyho/capstone_outputs/${DIR_DATE}
OUTPUT_DIR=$OUTPUT_ROOT/${SLURM_JOB_ID}_${MODEL}_${SIZE}_${L1}_${L2}_F${FINETUNE}_S${SPLIT}
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Environment Information"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "============================================================"

echo "============================================================"
echo "Running"
echo "============================================================"
uv run python model/main.py $L1 $L2 $OUTPUT_DIR --size $SIZE --model $MODEL --finetune $FINETUNE --split $SPLIT --quantize $QUANTIZE

echo "============================================================"
echo "Inference Complete"
echo "============================================================"
echo "End time: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================================"

# Remove checkpoint models from training
rm -r $OUTPUT_DIR/models/checkpoint*