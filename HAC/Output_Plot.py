import os
from rl_interaction.utils.Output_PlotterFiles import Plotter

algo_list = ['HAC', '']
steps = 3000
title = 'HAC'
path = 'pickle_files'
Plotter.plot_data(algo_list=algo_list, N=5, steps=steps, title=title, path=path)
