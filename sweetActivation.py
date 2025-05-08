from collections import defaultdict
from model import run_exp
from model import default_params as params
import utils as utl
from brian2 import Hz
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os

# ensure your compilers are set
os.environ['CC']  = '/opt/homebrew/bin/gcc-14'
os.environ['CXX'] = '/opt/homebrew/bin/g++-14'

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

# load ORN cell types → dict: {cell_type: [root_id,…]}
df_orn = pd.read_csv('Data/filtered_orn.csv')
cell_type_dict = defaultdict(list)
for _, row in df_orn.iterrows():
    cell_type_dict[row['primary_type']].append(int(row['root_id']))

# frequency sweep
freqs = list(range(20, 30, 2)) * Hz

# experiment names & file‐paths
exp_names = []
file_paths = []

# 1) sugar‐only controls
for f in freqs:
    params['r_poi'] = f
    name = f'sugar_only_{int(f/Hz)}Hz'
    run_exp(
        exp_name=name,
        neu_exc=neu_sugar,
        params=params,
        **config,
        force_overwrite=True
    )
    exp_names.append(name)
    file_paths.append(f'{config["path_res"]}/{name}.parquet')

# 2) sugar + each ORN cell type
for cell_type, orn_ids in cell_type_dict.items():
    for f in freqs:
        params['r_poi'] = f
        # combine sugar + this ORN population
        neu_combo = neu_sugar + orn_ids
        name = f'sugar_plus_{cell_type}_{int(f/Hz)}Hz'
        run_exp(
            exp_name=name,
            neu_exc=neu_combo,
            params=params,
            **config,
            force_overwrite=True
        )
        exp_names.append(name)
        file_paths.append(f'{config["path_res"]}/{name}.parquet')

# 3) load & compute rates
flyid2name = {nid: f'sugar_{i+1}' for i, nid in enumerate(neu_sugar)}
# (you may also want to map ORN IDs to labels if desired)

df_spike     = utl.load_exps(file_paths)
df_rate, df_std = utl.get_rate(
    df_spike,
    t_run=params['t_run'],
    n_run=params['n_run'],
    flyid2name=flyid2name
)

# save full matrices
root = config['path_res'].parent
df_rate.fillna(0).to_csv(root / 'all_rates.csv')
df_std.fillna(0).to_csv(root / 'all_rates_std.csv')

# 4) plot MN9 responses
mn9_ids = [
    720575940660219265,
    720575940618238523,
    720575940639332736  # add your third MN9 ID here
]

# subset to MN9 rows
df_plot = df_rate.loc[mn9_ids]

plt.figure(figsize=(8,6))
for mn in mn9_ids:
    # for each MN9, plot across all experiments
    plt.plot(
        [int(name.split('_')[-1].replace('Hz','')) for name in df_plot.columns],
        df_plot.loc[mn],
        marker='o',
        label=f'MN9 {mn}'
    )
plt.xlabel('Poisson rate (Hz)')
plt.ylabel('Firing rate (Hz)')
plt.title('MN9 activation vs. stimulus frequency')
plt.legend()
plt.tight_layout()
plt.show()