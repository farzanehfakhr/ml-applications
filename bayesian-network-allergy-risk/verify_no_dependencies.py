"""
A dependency-free sanity check for allergy_risk_bn.py: computes the same
CPTs by brute-force enumeration (law of total probability) instead of via
pgmpy, and checks the result against the write-up's worked examples.

Run this if you don't want to install pgmpy just to confirm the numbers.
"""

FS = {"High": 0.70, "Low": 0.30}
EE = {"Low": 0.80, "High": 0.20}
EP = {"Yes": 0.85, "No": 0.15}
ST = {"High": 0.75, "Low": 0.25}

AE_GIVEN_FS_EE = {
    ("High", "Low"): 0.02, ("High", "High"): 0.10,
    ("Low", "Low"): 0.20, ("Low", "High"): 0.60,
}

SR_GIVEN_AE_YES = {
    ("Yes", "High"): 0.05, ("Yes", "Low"): 0.10,
    ("No", "High"): 0.25, ("No", "Low"): 0.40,
}
SR_GIVEN_AE_NO = 0.01  # constant background risk, per the write-up


def p_severe_reaction(evidence=None):
    """
    evidence: optional dict with any subset of FoodSafety, EnvironmentalExposure,
    CarriesEpiPen, StaffTraining fixed; unset variables are marginalized out.
    """
    evidence = evidence or {}
    total = 0.0
    for fs, pfs in FS.items():
        if evidence.get("FoodSafety", fs) != fs:
            continue
        for ee, pee in EE.items():
            if evidence.get("EnvironmentalExposure", ee) != ee:
                continue
            p_ae_yes = AE_GIVEN_FS_EE[(fs, ee)]
            p_ae_no = 1 - p_ae_yes
            for ep, pep in EP.items():
                if evidence.get("CarriesEpiPen", ep) != ep:
                    continue
                for st, pst in ST.items():
                    if evidence.get("StaffTraining", st) != st:
                        continue
                    p_sr = p_ae_yes * SR_GIVEN_AE_YES[(ep, st)] + p_ae_no * SR_GIVEN_AE_NO
                    total += pfs * pee * pep * pst * p_sr

    # renormalize by the probability mass of the evidence itself
    norm = 0.0
    for fs, pfs in FS.items():
        if evidence.get("FoodSafety", fs) != fs:
            continue
        for ee, pee in EE.items():
            if evidence.get("EnvironmentalExposure", ee) != ee:
                continue
            for ep, pep in EP.items():
                if evidence.get("CarriesEpiPen", ep) != ep:
                    continue
                for st, pst in ST.items():
                    if evidence.get("StaffTraining", st) != st:
                        continue
                    norm += pfs * pee * pep * pst

    return total / norm


if __name__ == "__main__":
    checks = [
        ("Baseline (no evidence)", {}, 0.0194185),
        ("Example 1: FS=High,EE=Low,EP=Yes,ST=High", {
            "FoodSafety": "High", "EnvironmentalExposure": "Low",
            "CarriesEpiPen": "Yes", "StaffTraining": "High"}, 0.0108),
        ("Example 2: FS=Low,EE=High,EP=No,ST=Low", {
            "FoodSafety": "Low", "EnvironmentalExposure": "High",
            "CarriesEpiPen": "No", "StaffTraining": "Low"}, 0.244),
        ("Example 3: FS=High,EE=Low,EP=No,ST=High", {
            "FoodSafety": "High", "EnvironmentalExposure": "Low",
            "CarriesEpiPen": "No", "StaffTraining": "High"}, 0.0148),
    ]

    for label, evidence, expected in checks:
        actual = p_severe_reaction(evidence)
        status = "OK" if abs(actual - expected) < 1e-6 else "MISMATCH"
        print(f"[{status}] {label}: P(SevereReaction=Yes) = {actual:.6f} (expected {expected})")
