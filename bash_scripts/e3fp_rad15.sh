#!/bin/bash   
#SBATCH -J e3fp15                     # Job name
#SBATCH --time=48:00:00                 # Walltime                                      
#SBATCH --mem-per-cpu=16G             # memory/cpu (in MB) ### commented out              
#SBATCH --ntasks=1                      # 1 tasks                                               
#SBATCH --cpus-per-task=1           # number of cores per task                          
#SBATCH --gpus-per-task=1           # number of cores per task                          
#SBATCH --nodes=1                       # number of nodes                                       
#S BATCH --exclusive                     # node should not be shared with other jobs, only use this if you intend the node to be usable only by you as this will block other users from submitting jobs     to the same node                
#SBATCH --chdir=/vols/opig/users/raja # From where you want the job to be run
#SBATCH --mail-user=arun.raja@dtc.ox.ac.uk  # set email address                           
#SBATCH --mail-type=ALL                 # Spam us with everything, caution
#SBATCH --mail-type=begin               # Instead only email when job begins...
#SBATCH --mail-type=end                 # ... and ends
#S BATCH --partition=nagagpu04-high-debug  # Select a specific partition rather than default 
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=high-opig-test    # Select a specific partition rather than default
#S BATCH -w nagagpu04.cpu.stats.ox.ac.uk # Provide a specific node/nodelist rather than the standard nodelist associated with the partition (useful if you have a data setup on one specific node)
#SBATCH --output=/vols/opig/users/raja/slurm_outs/slurm_%j.out  # Writes standard output to this file. %j is jobnumber                             
#SBATCH --error=/vols/opig/users/raja/slurm_outs/slurm_%j.err   # Writes error messages to this file. %j is jobnumber
echo $CUDA_VISIBLE_DEVICES 
#python -m venv gin7
#source gin7/bin/activate
source /vols/opig/users/raja/miniconda3/etc/profile.d/conda.sh
#conda create --name gin_conda
conda activate gin_conda
echo "postera mlp"
python GDL-ActivityCliff-3D/e3fp_exp.py --dataset postera_sars_cov_2_mpro --model mlp --rad 1.5
echo "all proteins rad 1.5- rf"
python GDL-ActivityCliff-3D/e3fp_exp.py --dataset chembl_dopamine_d2 --model rf --rad 1.5 
python GDL-ActivityCliff-3D/e3fp_exp.py --dataset chembl_factor_xa --model rf --rad 1.5
# python GDL-ActivityCliff-3D/e3fp_exp.py --dataset postera_sars_cov_2_mpro --model rf --rad 1.5
echo "all proteins rad 1.5 - knn"
python GDL-ActivityCliff-3D/e3fp_exp.py --dataset chembl_factor_xa --model knn --rad 1.5
python GDL-ActivityCliff-3D/e3fp_exp.py --dataset chembl_dopamine_d2 --model knn --rad 1.5
# python GDL-ActivityCliff-3D/e3fp_exp.py --dataset postera_sars_cov_2_mpro --model knn --rad 1.5
echo "all proteins rad 1.5 - mlp"
python GDL-ActivityCliff-3D/e3fp_exp.py --dataset chembl_factor_xa --model mlp --rad 1.5
python GDL-ActivityCliff-3D/e3fp_exp.py --dataset chembl_dopamine_d2 --model mlp --rad 1.5
