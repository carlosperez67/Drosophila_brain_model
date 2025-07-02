import warnings

import pandas as pd
import numpy as np

##########
# analysis
def load_exps(l_pqt):
    '''Load simulation results from disk

    Parameters
    ----------
    l_pkl : list
        List of parquet files with simulation results

    Returns
    -------
    exps : df
        data for all experiments 'path_res'
    '''
    # cycle through all experiments
    dfs = []
    for p in l_pqt:
        # load metadata from pickle
        try:
            with open(p, 'rb') as f:
                df = pd.read_parquet(p)
                df.loc[:, 't'] = df.loc[:, 't'].astype(float)
                dfs.append(df)
        except Exception as e:
            warnings.warn(f"Skipping corrupted parquet file {p}: {e}")
            continue

    df = pd.concat(dfs)

    return df

def get_rate(df, t_run, n_run, flyid2name=dict()):
    '''Calculate rate and standard deviation for all experiments
    in df

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe generated with `load_exps` containing spike times
    t_run : float
        Trial duration in seconds
    n_run : int
        Number of trials
    flyid2name : dict (optional)
        Mapping between flywire IDs and custom names

    Returns
    -------
    df_rate : pd.DataFrame
        Dataframe with average firing rates
    df_std : pd.DataFrame
        Dataframe with standard deviation of firing rates
    '''

    rate, std, flyid, exp_name = [], [], [], []

    for e, df_e in df.groupby('exp_name', sort=False):
        for f, df_f in df_e.groupby('flywire_id'):

            r = np.zeros(n_run)
            for t, df_t in df_f.groupby('trial'):
                r[int(t)] = len(df_t) / t_run

            rate.append(r.mean())
            std.append(r.std())
            flyid.append(f)
            exp_name.append(e)

    d = {
        'r' : rate,
        'std': std,
        'flyid' : flyid,
        'exp_name' : exp_name,
    }
    df = pd.DataFrame(d)
    
    df_rate = df.pivot_table(columns='exp_name', index='flyid', values='r')
    df_std = df.pivot_table(columns='exp_name', index='flyid', values='std')
    
    if flyid2name:
        df_rate.insert(loc=0, column='name', value=df_rate.index.map(flyid2name).fillna(''))
        df_std.insert(loc=0, column='name', value=df_rate.index.map(flyid2name).fillna(''))

    return df_rate, df_std


def get_rate_for_neuron(
    df: pd.DataFrame,
    t_run: float,
    n_run: int,
    target_flyid: int
) -> np.ndarray:
    """
    Compute the firing rate (spikes/sec) of a single neuron for each trial.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing spike records with at least these columns:
          - 'trial'       : trial index (0 through n_run-1)
          - 'flywire_id'  : integer ID of the neuron
    t_run : float
        Duration of each trial in seconds.
    n_run : int
        Total number of trials.
    target_flyid : int
        Flywire ID of the neuron whose rate you want.

    Returns
    -------
    rates : np.ndarray
        1D array of length `n_run`.  Element `i` is the firing rate
        (spikes / t_run) observed in trial `i` for `target_flyid`.
        Trials with zero spikes will yield rate = 0.0.
    """
    # Initialize all trial‑rates to zero
    rates = np.zeros(n_run)

    # Filter for the neuron of interest
    df_neuron = df[df['flywire_id'] == target_flyid]

    r = np.zeros(n_run)

    for e, df_e in df_neuron.groupby('exp_name', sort=False):
        for f, df_f in df_e.groupby('flywire_id'):

            for t, df_t in df_f.groupby('trial'):
                r[int(t)] = len(df_t) / t_run


    return r
