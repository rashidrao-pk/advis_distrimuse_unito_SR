# srun   --partition=gracehopper   --nodes=1   --ntasks=1   --cpus-per-task=8   --gres=gpu:gh200:1   --mem=32G   --time=01:00:00   --pty bash

# srun -p epito --gres=gpu:a100:1 -J "ADVIS" --time=06:00:00 --pty bash
source /beegfs/home/mrashid/pt_312/bin/activate
export PYTHONPATH=/opt/pytorch-v2.7.1/lib/python3.12/site-packages
export PROJECT_DIR=/beegfs/home/mrashid/repos/DistriMuSe/advis_distrimuse_unito_SR
cd "$PROJECT_DIR"