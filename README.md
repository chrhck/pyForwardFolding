# pyForwardFolding

[![ci](https://github.com/chrhck/pyForwardFolding/workflows/ci/badge.svg)](https://github.com/chrhck/pyForwardFolding/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Description

**Welcome!**

## Quickstart

[Install Hatch](https://hatch.pypa.io/latest/install/), then install the project:

```sh
❯ cd path/to/repo
# Try running the tests
❯ hatch run coverage run
```

## Model Structure

We have a collection of MC events, with true properties $X = (E, \ldots)$ , reconstructed quantities $\hat{X} = (\hat{E}, \ldots)$ and a weight $w$.
The weight encodes the instrument response of the detector and is defined such that the rate $R$ in Hz is given by: 

```math
R(X) = w \cdot M(X)
```

where $M(X \mid \theta)$ is the differential flux model:

```math
M(X \mid \theta)=\frac{\partial^2 \Phi(X, \theta)}{\partial E \partial \Omega}
```

A model can generally consist of multiple components, with $`\theta = \{ \theta_C \}_C `$

```math
M(X \mid \theta) = \sum_C m_C(X \mid \theta_C)
```

For better composability, we can write a model component as a product (for example, a powerlaw with an exponential cutoff) of factors ($F$)

```math
m_C(X \mid \theta_C) = \prod_f F_{f, C}(X \mid \theta_f) ,
```

where $f$ indexes the factors.

Events (with index $i$) are binned according to their reconstructed quantities, where the expectation $\mu_A$ (weight) in bin A is given by:

```math
\mu_A(X, \hat{X}, \theta) = \sum_i^{N} I_{A}(\hat{X}_{i}) \cdot w_i \cdot M(X \mid \theta ) = \sum_i^{N} I_{A}(\hat{X}_i) \cdot w_i \cdot \sum_j m_j(X \mid \theta_C) = \sum_i^{N} I_{A}(\hat{X}_i) \cdot \sum_C m_C(X \mid \theta_C) \cdot w_i
```

where the Indicator $I_{A}$ is $1$ if the event belongs to bin A, and $0$ otherwise.



### Model Parameters

In general, models will have free parameters of interest $\theta$, the simplest being the model normalization $\Phi_0$. Additionally there might be nuisance parameters $\nu$, which are typically profiled over in the analysis.

### Detector Systematics

Detector systematics $\hat{\nu}$, such as the detection efficiency, modify the instrument response and thus the weight $w$. They can either be parametrized on an event-by-event level:

```math
w \to w \cdot S(X)
```

or on a bin-by-bin level:

```math
\hat{\mu}_A(X, \hat{X}, \theta) = S_A \cdot \mu_A(X, \hat{X}, \theta) = S_A \cdot \sum_i^{N} I_{A}\left(\hat{X}_i \right) \cdot \sum_C m_C(X \mid \theta_C) \cdot w_i
```

### Likelihood

The (binned) likelihood is constructed by treating each bin independently:

```math
\mathcal{L}\left (\theta \mid \mathrm{data} \right) = \prod_A \mathcal{L_A}(\hat{\mu}_A(X, \hat{X}, \theta) , \mathrm{data})
```

### Multiple Detectors

When combining multiple detectors (or in general different histograms), we use the same model $M$ but different weights $w$, datasets $X, \hat{X}$ and potentially different detector systematics.
The most general case would look like this:

```math
\hat{\mu}^D_A(X^D, \hat{X^D}, \theta) ~= S^D_A \cdot \sum_i^{N^D} I^D_{A}\left(\hat{X}^D_i \right) \cdot M(X^D \mid \theta) \cdot w_i^D \cdot S^D(X^D) = S^D_A \cdot \sum_i^{N^D} I^D_{A}\left(\hat{X}^D_i \right) \cdot \sum_C m_C(X^D \mid \theta_C) \cdot w_i^D \cdot S^D(X^D) = S^D_A \cdot \sum_i^{N^D} I^D_{A}\left(\hat{X}^D_i \right) \cdot \sum_C \prod_f \left(F_{f, C}(X \mid \theta_f) \right) \cdot w_i^D \cdot S^D(X^D)
```

Representing the per-event-systematics as model-factors:

```math
\hat{\mu}^D_A(X^D, \hat{X^D}, \theta)  ~= S^D_A \cdot \sum_i^{N^D} I^D_{A}\left(\hat{X}^D_i \right) \cdot \sum_C \prod_f F_{f, C}(X \mid \theta_f) \cdot w_i^D
```

## Class Structure

The model structure above is mapped to classes in the following manner:
| Model Part                              | Class               | Rationale                                                                                                                                              |
| ----------------------------------------| --------------------| -------------------------------------------------------------------------------------------------------------------------------------------------------|
| $F(X \mid \theta_f)$                    | `Factor`            | Factor of a model component                                                                                                                            |
| $m_C(X^D) \cdot w_i^D \cdot S^D(X^D)$   | `ModelComponent`    | For better composability, `ModelComponent` is a collection of `Factor`'s. This also allows for the implementation of per-event nuisance factors.       |
| $M^D(X^D \mid \theta)\cdot w^D_i $      | `Model`             | Sums over model components                                                                                                                             |
| $\mu^D_A(X^D, \hat{X^D}, \theta)$       | `BinnedExpectation` | Creates a histogram                                                                                                                                    |
| $S_A^D$                                 | `BinnedFactor`      | An additive factor for a histogram                                                                                                                |
| $\hat{\mu}^D_A(X^D, \hat{X^D}, \theta)$ | `Analysis`          | Combines `BinnedExpectations` with `BinnedFactors`                                                                                          | 
| $\mathcal{L}$                           | `Likelihood`        | Evaluates likelihood for `Analysis`                                                                                                                    |


## Further information

See [CONTRIBUTING.md](.github/CONTRIBUTING.md).
