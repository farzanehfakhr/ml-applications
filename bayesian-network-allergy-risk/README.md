# Bayesian Network: Allergy Risk Assessment for School Trips

A Bayesian network that estimates the risk of a severe allergic reaction for a student on an outdoor school trip, and supports "what-if" scenario analysis for trip planning. Built for MET CS 767 (Boston University), Assignment 6.

## The scenario

A school nurse, trip coordinator, or parent wants to assess how safe an outdoor trip is for a student with allergies, and see how that risk changes under different preparation levels — poor vs. strict food-safety procedures, trained vs. untrained staff, with or without an EpiPen.

## Network structure

```
FoodSafety ------------\
                         v
EnvironmentalExposure -> AllergenExposure -\
                                              v
                  CarriesEpiPen --------> SevereReaction
                  StaffTraining ---------/
```

Four observable inputs (`FoodSafety`, `EnvironmentalExposure`, `CarriesEpiPen`, `StaffTraining`) feed into two inferred variables: whether the student is exposed to an allergen (`AllergenExposure`), and whether that leads to a severe reaction (`SevereReaction`).

**Conditional probability tables** (full values in `allergy_risk_bn.py`):

| FS | EE | P(AE=Yes) |
|---|---|---|
| High | Low | 0.02 |
| High | High | 0.10 |
| Low | Low | 0.20 |
| Low | High | 0.60 |

| AE | EP | ST | P(SR=Yes) |
|---|---|---|---|
| Yes | Yes | High | 0.05 |
| Yes | Yes | Low | 0.10 |
| Yes | No | High | 0.25 |
| Yes | No | Low | 0.40 |
| No | any | any | 0.01 (background risk) |

## Results

| Scenario | Evidence | P(Severe Reaction) |
|---|---|---|
| Baseline (no evidence) | — | 1.94% |
| Example 1: well-prepared trip | Good food safety, low exposure, has EpiPen, trained staff | **1.08%** |
| Example 2: poorly-prepared trip | Poor food safety, high exposure, no EpiPen, untrained staff | **24.4%** (22x Example 1) |
| Example 3: safe trip, forgot EpiPen | Same as Example 1 but no EpiPen | **1.48%** (37% higher than Example 1) |

A sensitivity analysis on the baseline scenario found predicted risk scales roughly proportionally with the `AllergenExposure` probability — a 2%→20% change in exposure probability (holding everything else fixed) raises severe-reaction risk from 1.08% to 1.8%, confirming food-safety procedures are the highest-leverage variable in the model.

## Why a Bayesian network (not a neural net or genetic algorithm)

The write-up also argues the case for the chosen model type, briefly:

- **Vs. a neural network:** no large labeled dataset of past trips exists (severe allergic reactions are rare, privacy-protected, and inconsistently documented), so a BN's ability to work from expert-specified probabilities rather than training data is a practical necessity here. A BN also natively supports partial evidence and "what-if" queries on internal variables; a standard feed-forward net gives one forward prediction per fixed input and doesn't naturally do either.
- **Vs. a genetic algorithm:** a GA is an optimizer, not a probabilistic model — it has no native way to represent conditional dependencies or compute a posterior probability, so it isn't a fit for this task's actual requirement (quantitative risk estimation under partial evidence).

## Scaling to a real system

The write-up discusses what a production version would need: real incident data to replace the current expert-elicited CPTs (complicated by HIPAA/FERPA and how rare these incidents are); location- and season-specific parameters, since food safety and allergen exposure vary widely by district and region; richer, non-binary variables (a spectrum of food-safety quality, multiple EpiPen-availability states); a dynamic/temporal model, since the current network treats a whole trip as one static time block; and real operational and regulatory hurdles around integrating with school nurse workflows and handling protected student medical data.

## Files

| File | Description |
|---|---|
| `allergy_risk_bn.py` | The network built with [pgmpy](https://pgmpy.org/) (`pip install pgmpy`); exact inference via `VariableElimination` |
| `verify_no_dependencies.py` | A dependency-free re-implementation (plain Python, brute-force enumeration) that checks all four results above match exactly — no install required to confirm the numbers |

Both scripts reproduce the original write-up's results exactly (verified: baseline 0.019419, and all three worked examples match to 4+ decimal places).

## Tools

Bayesian networks, probabilistic inference (variable elimination), pgmpy, Python

---
*Completed as part of graduate coursework (METCS767, Machine Learning) at Boston University.*
