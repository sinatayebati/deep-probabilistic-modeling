# Pyro-deep-probabilistic-models
This repository includes various probabilistic models developed based on Pyro, a deep universal probabilistic programming framework backed by PyTorch.


![](images/pyro_logo.png)

Probability is the mathematics of reasoning under uncertainty, much as calculus is the mathematics for reasoning about rates of change. It provides a unifying theoretical framework for understanding much of modern machine learning and AI: models built in the language of probability can capture complex reasoning, know what they do not know, and uncover structure in data without supervision.

Specifying probabilistic models directly can be cumbersome and implementing them can be very error-prone. Probabilistic programming languages (PPLs) solve these problems by marrying probability with the representational power of programming languages. A probabilistic program is a mix of ordinary deterministic computation and randomly sampled values representing a generative process for data.

By observing the outcome of a probabilistic program, we can describe an inference problem, roughly translated as: “what must be true if this random choice had a certain observed value?” PPLs explicitly enforce a separation of concerns already implicit in the mathematics of probability between the specification of a model, a query to be answered, and an algorithm for computing the answer.

Pyro is a probabilistic programming language built on Python and PyTorch. Pyro programs are just Python programs, while its main inference technology is stochastic variational inference, which converts abstract probabilistic computations into concrete optimization problems solved with stochastic gradient descent in PyTorch, making probabilistic methods applicable to previously intractable model and dataset sizes.

In this repository, we take a brief, opinionated tour of the basic concepts of probabilistic machine learning and probabilistic programming with Pyro. We do so via an example data analysis problem involving linear regression, one of the most common and basic tasks in machine learning. We will see how to use Pyro’s modeling language and inference algorithms to incorporate uncertainty into estimates of regression coefficients.


# Setup and Requirements

This repository has been tested on the following environemt:

```
pyro == 1.8.4
pytorch == 2.0.0
pandas == 2.0.1
numpy == 1.24.3
matplotlib == 3.7.1
seaborn == 0.12.2
graphviz == 0.20.1
```
Clone the repository in your local machine, and activate your virtual environement:

```
git clone https://github.com/sina-tyi/Deep-probabilistic-modeling.git
cd Deep-probabilistic-modeling
source PyroVenv/bin/activate
```

Install the requirements:

```
pip install -r requirements.txt
```