#!/bin/bash
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=ConInt
#SBATCH --time=36:00:00
#SBATCH --account=pi-lgagliardi


../SHARC_PYSCF3.py QM.in >> QM.log 2>> QM.err
err=$?

exit $err
