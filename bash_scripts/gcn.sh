#!/bin/bash   
#SBATCH -J gcn_mlp                     # Job name
#SBATCH --time=12:00:00                 # Walltime                                      
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
#SBATCH --clusters=all
#S BATCH -w nagagpu04.cpu.stats.ox.ac.uk # Provide a specific node/nodelist rather than the standard nodelist associated with the partition (useful if you have a data setup on one specific node)
#SBATCH --output=/vols/opig/users/raja/slurm_outs/slurm_%j.out  # Writes standard output to this file. %j is jobnumber                             
#SBATCH --error=/vols/opig/users/raja/slurm_outs/slurm_%j.err   # Writes error messages to this file. %j is jobnumber
echo $CUDA_VISIBLE_DEVICES 
source /vols/opig/users/raja/miniconda3/etc/profile.d/conda.sh
conda activate gin_conda
echo "dataset: $1";
echo "model: $2";
python GDL-ActivityCliff-3D/gcn_exp.py --dataset $1 --model $2 >> GDL-ActivityCliff-3D/terminal_output/gcn_$1_$2.txt
#python GDL-ActivityCliff-3D/gin_exp.py --dataset postera_sars_cov_2_mpro --model rf >> GDL-ActivityCliff-3D/terminal_output/gin_rf.txt
#python GDL-ActivityCliff-3D/gin_exp.py --dataset postera_sars_cov_2_mpro --model knn >> GDL-ActivityCliff-3D/terminal_output/gin_knn2.txt
#python GDL-ActivityCliff-3D/gin_exp.py --dataset postera_sars_cov_2_mpro --model mlp >> GDL-ActivityCliff-3D/terminal_output/gin_mlp.txt
