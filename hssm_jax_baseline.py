"""
This script uses GPU acceleration offered in HSSM
To use the acceleration, install it like this:
pip install --upgrade "jax[cuda12]"
pip install hssm

To check if GPU is used with JAX (only works on compute servers wiesel and ramones), try:
import jax
print(jax.default_backend())  # Should output 'gpu'
print(jax.devices())  # Should list GPU devices

"""

import pandas as pd
import hssm
import os
import arviz as az
import numpy as np 
import seaborn as sns
import xarray as xr
import json 
import pickle
import os 
import matplotlib.pyplot as plt

base_dir = "/data/p_02956/EXNAT_B/DDM/EXNAT-1_HSSM"


# load data
data_file = os.path.join(base_dir, "Data", "df_single_nback_hssm.csv")
df_single_nback = pd.read_csv(data_file)

# center age
#data["age_centered"] = data["age"] - data["age"].mean()

model_baseline = hssm.HSSM(
    data=df_single_nback,
    model="ddm",
    prior_settings="safe",
)

baseline_graph = model_baseline.graph()
baseline_graph.render(os.path.join(base_dir, "baseline_model/bm_diagnostics/baseline_graph.pdf"))

# model sample
samples_baseline = model_baseline.sample(
    draws=8000,
    tune=1000,
    # burn=3000,
    chains=4,
    sampler="nuts_numpyro",
    target_accept=0.9,
    #random_seed=11,
)

# save model build in function 
model_baseline.save_model(os.path.join(base_dir, "baseline_model"))

# save statistics as csv file 
baseline_stats = az.summary(model_baseline.traces, var_names=["a", "t", "z", "v"])
baseline_stats.to_csv(os.path.join(base_dir, "baseline_model/bm_diagnostics/baseline_stats_summary.csv"))

# plot the traces
plt.figure()
model_baseline.plot_trace()
plt.savefig(os.path.join(base_dir, "baseline_model/bm_diagnostics/baseline_model_traces.pdf"))
plt.close()
print("traces have been plotted and saved")

# plot posterior predictives 
plt.figure()
ax = hssm.plotting.plot_posterior_predictive(model_baseline)
sns.despine()
ax.set_ylabel("")
plt.title("Posterior Predictive Plot")
plt.savefig(os.path.join(base_dir, "baseline_model/bm_diagnostics/baseline_model_posterior_predictie.pdf"))
plt.close()
print("posterior predictives have been plotted and saved")

# save sampling
model_baseline.traces.to_netcdf(os.path.join(base_dir, "baseline_posterior.nc"))
print("traces have been converted to nc")

plt.figure()
az.plot_pair(
    model_baseline.traces,
    var_names=["v", "t", "z", "a"],
    divergences=True
)
plt.savefig(os.path.join(base_dir, "baseline_model/bm_diagnostics/baseline_plot_pair.pdf"))
plt.close()
print("plot pairs have been plotted and saved")

# saving traces and posteriors separately 
traces = model_baseline.traces
posterior = traces.posterior 

df_posterior = posterior.to_dataframe()
df_traces = traces.to_dataframe()

df_posterior.to_csv(os.path.join(base_dir, "baseline_model/bm_diagnostics/baseline_posterior_data.csv"))
df_traces.to_csv(os.path.join(base_dir, "baseline_model/bm_diagnostics/baseline_traces.csv"))
print("script is done")
