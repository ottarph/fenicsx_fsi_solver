#!/bin/bash
# Run the test suite on one eX3 node. Submit from the repo root: sbatch jobs/tests.sh
#
# pytest runs on a single rank; the restart tests spawn their own `mpiexec -n 2`,
# so the job asks for 4 physical cores.
#SBATCH -J xfsi-tests
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH -t 02:00:00
#SBATCH -o /home/hellanottar/xfsi/logs/tests_%j.txt

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Hydra mpiexec inside a Slurm job otherwise launches ranks through srun; keep them on this node
export HYDRA_BOOTSTRAP=fork

cd /home/hellanottar/code/xfsi_solver
~/miniforge3/bin/conda run --no-capture-output -n xfsi_solver python -u -m pytest -v "$@"
