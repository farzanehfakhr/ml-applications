"""
A Bayesian network estimating the risk that a student with allergies
experiences a severe allergic reaction during a school outdoor trip,
given evidence about trip conditions and student preparedness.

Network structure:
    FoodSafety -----------\\
                            v
    EnvironmentalExposure -> AllergenExposure -\\
                                                  v
                      CarriesEpiPen ---------> SevereReaction
                      StaffTraining ----------/

Built for MET CS 767 (Boston University), Assignment 6. The original
write-up computed these probabilities by hand (worked examples + a
sensitivity analysis) and, separately, via a query tool whose output
format (pgmpy's VariableElimination) appears in the submitted screenshots -
this script is a reconstruction of that query tool using the CPTs stated
explicitly in the write-up's tables, built to verify and reproduce those
hand-computed results, not transcribed from a captured code cell (the
model-construction cell itself wasn't among the captured screenshots).

Requires: pip install pgmpy
"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# ----------- Network structure -----------
model = DiscreteBayesianNetwork([
    ("FoodSafety", "AllergenExposure"),
    ("EnvironmentalExposure", "AllergenExposure"),
    ("AllergenExposure", "SevereReaction"),
    ("CarriesEpiPen", "SevereReaction"),
    ("StaffTraining", "SevereReaction"),
])

# ----------- Root node priors -----------
cpd_fs = TabularCPD("FoodSafety", 2, [[0.70], [0.30]],
                     state_names={"FoodSafety": ["High", "Low"]})

cpd_ee = TabularCPD("EnvironmentalExposure", 2, [[0.20], [0.80]],
                     state_names={"EnvironmentalExposure": ["High", "Low"]})

cpd_ep = TabularCPD("CarriesEpiPen", 2, [[0.85], [0.15]],
                     state_names={"CarriesEpiPen": ["Yes", "No"]})

cpd_st = TabularCPD("StaffTraining", 2, [[0.75], [0.25]],
                     state_names={"StaffTraining": ["High", "Low"]})

# ----------- AllergenExposure | FoodSafety, EnvironmentalExposure -----------
# P(AE=Yes | FS, EE):
#   FS=High, EE=High -> 0.10      FS=High, EE=Low -> 0.02
#   FS=Low,  EE=High -> 0.60      FS=Low,  EE=Low -> 0.20
cpd_ae = TabularCPD(
    "AllergenExposure", 2,
    [
        [0.10, 0.02, 0.60, 0.20],  # P(AE=Yes | FS,EE) in column order below
        [0.90, 0.98, 0.40, 0.80],  # P(AE=No  | FS,EE)
    ],
    evidence=["FoodSafety", "EnvironmentalExposure"],
    evidence_card=[2, 2],
    state_names={
        "AllergenExposure": ["Yes", "No"],
        "FoodSafety": ["High", "Low"],
        "EnvironmentalExposure": ["High", "Low"],
    },
)

# ----------- SevereReaction | AllergenExposure, CarriesEpiPen, StaffTraining -----------
# When AE=Yes, risk depends on EpiPen + StaffTraining:
#   EP=Yes, ST=High -> 0.05   EP=Yes, ST=Low -> 0.10
#   EP=No,  ST=High -> 0.25   EP=No,  ST=Low -> 0.40
# When AE=No, a constant ~1% background risk is used regardless of EP/ST
# (as stated in the write-up's worked examples).
cpd_sr = TabularCPD(
    "SevereReaction", 2,
    [
        # AE=Yes                              AE=No
        [0.05, 0.10, 0.25, 0.40, 0.01, 0.01, 0.01, 0.01],  # P(SR=Yes | AE,EP,ST)
        [0.95, 0.90, 0.75, 0.60, 0.99, 0.99, 0.99, 0.99],  # P(SR=No  | AE,EP,ST)
    ],
    evidence=["AllergenExposure", "CarriesEpiPen", "StaffTraining"],
    evidence_card=[2, 2, 2],
    state_names={
        "SevereReaction": ["Yes", "No"],
        "AllergenExposure": ["Yes", "No"],
        "CarriesEpiPen": ["Yes", "No"],
        "StaffTraining": ["High", "Low"],
    },
)

model.add_cpds(cpd_fs, cpd_ee, cpd_ep, cpd_st, cpd_ae, cpd_sr)
assert model.check_model()

infer = VariableElimination(model)


def query_severe_reaction(evidence):
    """evidence e.g. {"FoodSafety": "High", "EnvironmentalExposure": "Low",
                       "CarriesEpiPen": "Yes", "StaffTraining": "High"}"""
    return infer.query(variables=["SevereReaction"], evidence=evidence, show_progress=False)


if __name__ == "__main__":
    print("=== Baseline: P(SevereReaction) ===")
    print(infer.query(variables=["SevereReaction"], show_progress=False))
    # Expected: P(SR=Yes) ~= 0.0194

    print("\n=== Example 1: High safety scenario ===")
    print("Evidence: FS=High, EE=Low, EP=Yes, ST=High")
    print(query_severe_reaction({
        "FoodSafety": "High", "EnvironmentalExposure": "Low",
        "CarriesEpiPen": "Yes", "StaffTraining": "High",
    }))
    # Expected: P(SR=Yes) ~= 0.0108 (~1.08%)

    print("\n=== Example 2: Low safety scenario ===")
    print("Evidence: FS=Low, EE=High, EP=No, ST=Low")
    print(query_severe_reaction({
        "FoodSafety": "Low", "EnvironmentalExposure": "High",
        "CarriesEpiPen": "No", "StaffTraining": "Low",
    }))
    # Expected: P(SR=Yes) ~= 0.244 (~24.4%) -- over 20x the Example 1 risk

    print("\n=== Example 3: Safe conditions, but no EpiPen ===")
    print("Evidence: FS=High, EE=Low, EP=No, ST=High")
    print(query_severe_reaction({
        "FoodSafety": "High", "EnvironmentalExposure": "Low",
        "CarriesEpiPen": "No", "StaffTraining": "High",
    }))
    # Expected: P(SR=Yes) ~= 0.0148 (~1.48%)
