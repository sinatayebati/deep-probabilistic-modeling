import logging
import os

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz 

import pyro

DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
df = data[["cont_africa", "rugged", "rgdppc_2000"]]

df = df[np.isfinite(df.rgdppc_2000)]
df["rgdppc_2000"] = np.log(df["rgdppc_2000"])

train = torch.tensor(df.values, dtype=torch.float)
is_cont_africa, ruggedness, log_gdp = train[:, 0], train[:, 1], train[:, 2]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
african_nations = df[df["cont_africa"] == 1]
non_african_nations = df[df["cont_africa"] == 0]
sns.scatterplot(x=non_african_nations["rugged"],
                y=non_african_nations["rgdppc_2000"],
                ax=ax[0])
ax[0].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="Non African Nations")
sns.scatterplot(x=african_nations["rugged"],
                y=african_nations["rgdppc_2000"],
                ax=ax[1])
ax[1].set(xlabel="Terrain Ruggedness Index",
          ylabel="log GDP (2000)",
          title="African Nations");


# maximum-likelihood linear regression

"""
If we write out the formula for our linear regression predictor
BX + a
as a Python expression, we get the following:
"""
import pyro.distributions as dist
import pyro.distributions.constraints as constraints

def simple_model(is_cont_africa, ruggedness, log_gdp=None):
    a = pyro.param("a", lambda: torch.randn(()))
    b_a = pyro.param("bA", lambda: torch.randn(()))
    b_r = pyro.param("bR", lambda: torch.randn(()))
    b_ar = pyro.param("bAR", lambda: torch.randn(()))
    sigma = pyro.param("sigma", lambda: torch.ones(()), constraint=constraints.positive)
    
    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness

    with pyro.plate("data", len(ruggedness)):
        return pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)
    
pyro.render_model(simple_model, model_args=(is_cont_africa, ruggedness, log_gdp),
                   render_distributions=True, render_params=True)

# from maximum-likelihood regression to Bayesian regression

def model(is_cont_africa, ruggedness, log_gdp=None):
    a = pyro.sample("a", dist.Normal(0., 10.))
    b_a = pyro.sample("bA", dist.Normal(0., 1.))
    b_r = pyro.sample("bR", dist.Normal(0., 1.))
    b_ar = pyro.sample("b_AR", dist.Normal(0., 1.))
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))

    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness

    with pyro.plate("data", len(ruggedness)):
        return pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)
    
pyro.render_model(model, model_args=(is_cont_africa, ruggedness, log_gdp), render_distributions=True)


# mean-field variational approximation for Bayesian linear regression in Pyro

"""
Background: “guide” programs as flexible approximate posteriors
In variational inference, we introduce a parameterized distribution 
 to approximate the true posterior, where 
 are known as the variational parameters. This distribution is called the variational
 distribution in much of the literature, and in the context of Pyro it’s 
 called the guide (one syllable instead of nine!).

 Just like the model, the guide is encoded as a Python program guide()
 that contains pyro.sample and pyro.param statements.
 It does not contain observed data, since the guide needs to be a properly
 normalized distribution so that it is easy to sample from. Note that Pyro enforces
 that model() and guide() should take the same arguments.

 Allowing guides to be arbitrary Pyro programs opens up the possibility of
 writing guide families that capture more of the problem-specific structure of
 the true posterior, expanding the search space in only helpful directions,
 as depicted schematically in the figure below.
"""

def custom_guide(is_cont_africa, ruggedness, log_gdp=None):
    a_loc = pyro.param('a_loc', lambda: torch.tensor(0.))
    a_scale = pyro.param('a_scale', lambda: torch.tensor(1.),
                         constraint=constraints.positive)
    sigma_loc = pyro.param('sigma_loc', lambda: torch.tensor(1.),
                           constraint=constraints.positive)
    weights_loc = pyro.param('weights_loc', lambda:torch.randn(3))
    weights_scale = pyro.param('weights_scale', lambda: torch.ones(3),
                               constraint=constraints.positive)
    a = pyro.sample("a", dist.Normal(a_loc, a_scale))
    b_a = pyro.sample("bA", dist.Normal(weights_loc[0], weights_scale[0]))
    b_r = pyro.sample("bR", dist.Normal(weights_loc[1], weights_scale[1]))
    b_ar = pyro.sample("bAR", dist.Normal(weights_loc[2], weights_scale[2]))
    sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))
    return {"a":a, "b_a":b_a, "b_r":b_r, "b_ar":b_ar, "sigma":sigma}

pyro.render_model(custom_guide, model_args=(is_cont_africa, ruggedness, log_gdp),render_params=True)

"""
Pyro also contains an extensive collection of “autoguides” which automatically
 generate guide programs from a given model. Like our handwritten guide,
 all pyro.autoguide.AutoGuide instances (which are themselves just functions
 that take the same arguments as the model) return a dictionary of values for
 each pyro.sample site they contain.

 The simplest autoguide class is AutoNormal, which automatically generates a guide
 in a single line of code that is equivalent to the one we wrote out by hand above:
"""

auto_guide = pyro.infer.autoguide.AutoNormal(model)