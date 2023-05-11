# import packages

# general tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse

from modules import *

# python visualise.py --target chembl_dopamine_d2 --task_x ac_test --metric_x MCC --task_y qsar_test --metric_y MAE
# python visualise.py --target postera_sars_cov_2_mpro --task_x ac_test --metric_x MCC --task_y qsar_test --metric_y MAE
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, help='Description of arg1')
parser.add_argument('--task_x', type=str, help='Description of arg1')
parser.add_argument('--metric_x', type=str, help='Description of arg2')
parser.add_argument('--task_y', type=str, help='Description of arg2')
parser.add_argument('--metric_y', type=str, help='Description of arg2')

args = parser.parse_args()

target = args.target
task_x = args.task_x
metric_x = args.metric_x
task_y = args.task_y
metric_y = args.metric_y

visualise_results(target,
                      task_x,
                      metric_x,
                      task_y,
                      metric_y,
                      decimals_mean = 6,
                      decimals_std = 6,
                      plot_legend = True,
                      legend_loc = "upper right",
                      plot_title = True,
                      plot_x_label = True,
                      plot_y_label = True,
                      plot_x_ticks = True,
                      plot_y_ticks = True,
                      x_axis_units = "",
                      y_axis_units = "",
                      plot_error_bars = True,
                      x_tick_stepsize = "auto",
                      y_tick_stepsize = "auto",
                      xlim = None,
                      ylim = None,
                      size = 12, 
                      linear_regression = False, 
                      filepath_to_save = "results/"+target+"/plots/"+task_x + "_"+ metric_x +"_"+ task_y +"_"+ metric_y+"_scatter.svg")