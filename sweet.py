# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:20:26 2023

@author: lijin
"""

from model import run_exp
from model import default_params as params
import utils as utl
from brian2 import Hz
import pickle
from pathlib import Path

# configuration
config = {
    'path_comp': './Completeness_783.csv',
    'path_con':  './Connectivity_783.parquet',
    'n_proc':   -1,
    'path_res': Path('./sweet/GRN')
}

# load sugar GRN IDs
sweet_df   = pd.read_csv('Data/sweet.csv')
neu_sugar  = [int(i) for i in sweet_df['root_id']]

# freqs =  [ *range(10, 201, 10) ]
freqs = [10]
# run siumulations
# Calculate which neurons respond to sugar GRN firing at the specified frequencies.
for f in freqs:

    params['r_poi'] = f * Hz

    run_exp(exp_name='sugarR_{}Hz'.format(f), neu_exc=neu_sugar, params=params, **config, force_overwrite=True)

# process result
# dictionary containing the GRN names and each Flywire ID.
flyid2name = { f: 'sugar_{}'.format(i+1) for i, f in enumerate(neu_sugar) }

# list of output files
ps = ['{}/sugarR_{}Hz.parquet'.format(config['path_res'], f) for f in freqs ]

# load data from disk
df_spike = utl.load_exps(ps)
df_rate, df_rate_std = utl.get_rate(df_spike, t_run=params['t_run'], n_run=params['n_run'], flyid2name=flyid2name)

# save the spiking rates and standard deviations for each neuron
df_rate.fillna(0).to_csv(config['path_res'].parent / 'sweet_rate.csv')
df_rate_std.fillna(0).to_csv(config['path_res'].parent / 'sweet_rate_std.csv')