#!/bin/bash
#SBATCH --job-name=downsample
#SBATCH --output=downsample_%j.out
#SBATCH --error=downsample_%j.err
#SBATCH --account=IscrC_DATIV-ML   
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=160G


module purge

# Caricamento moduli necessari (se servono su Leonardo)
module load python/3.11

# Attivazione dell'ambiente virtuale dove hai astropy e numpy
source /leonardo/home/userexternal/gvitanza/.venv/bin/activate

# Esecuzione dello script Python
python downsampling.py