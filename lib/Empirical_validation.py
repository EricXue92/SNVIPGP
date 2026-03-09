import seaborn as sns
sns.set(style="white", font_scale=1.5)

import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

# Reproducibility
np.random.seed(0)

# --------------------------
# 1. Load and sanity-check data
# --------------------------
with open("../ood_leakage.pkl", "rb") as f:
    data = pickle.load(f)

ind_prob = data["iid"]["predictive_probs"]
ind_label = data["iid"]["predictive_labels"]
ind_unc = data["iid"]["predictive_uncertainty"]

ood_prob = data["ood"]["predictive_probs"]
ood_unc = data["ood"]["predictive_uncertainty"]

# Sanity checks
assert ind_unc.shape[0] == ind_prob.shape[0] == ind_label.shape[0], \
    f"InD length mismatch"
assert ood_unc.shape[0] == ood_prob.shape[0], \
    f"OOD length mismatch"

# --------------------------
# 2. Compute ROC curve for OOD detector
# --------------------------
# Label: 0=InD, 1=OOD; Score: uncertainty (higher → more likely OOD)
y_true = np.concatenate([np.zeros_like(ind_unc), np.ones_like(ood_unc)])
y_score = np.concatenate([ind_unc, ood_unc])
fpr_roc, tpr_roc, thresholds = roc_curve(y_true, y_score)

# Notation alignment with theory:
# - FPR (from ROC) = δ (False Rejection Rate of InD)
# - TPR (from ROC) = 1-γ (True detection rate of OOD)
# - Therefore: γ = 1 - TPR (False Acceptance Rate)

# Compute π (OOD proportion in NEW notation)
pi_ood = len(ood_unc) / (len(ind_unc) + len(ood_unc))  # P(OOD) = π
pi_ind = 1 - pi_ood  # P(InD) = 1-π

alpha = 0.01  # miscoverage level (target coverage = 99%)


# --------------------------
# 3. Conformal helper functions
# --------------------------
def predict_set(probs: np.ndarray, alpha: float):
    """Return conformal prediction sets for each row of `probs`."""
    idx = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, idx, axis=1)
    cumsums = np.cumsum(sorted_probs, axis=1)
    cutoffs = (cumsums < (1 - alpha)).sum(axis=1)
    return [idx[i, : cutoffs[i] + 1] for i in range(len(probs))]


def theoretical_coverage_worst_case(delta: float, gamma: float, pi: float, alpha: float):
    numerator = (1 - pi) * (1 - delta) * (1 - alpha)
    denominator = (1 - pi) * (1 - delta) + pi * gamma
    return numerator / denominator if denominator > 0 else 0.0

fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# ------------------------------------------------------------
# Panel (a): Coverage gap vs. FRR (δ)
# ------------------------------------------------------------
delta_targets = np.linspace(0, 0.8, 9)  # FRR = FPR in ROC
empirical_gaps_a = []
theoretical_gaps_a = []
delta_actual = []

for target_delta in delta_targets:
    # Find closest FPR in ROC (FPR = δ)
    i = np.argmin(np.abs(fpr_roc - target_delta))
    thr = thresholds[i]
    delta = fpr_roc[i]
    gamma = 1 - tpr_roc[i]  # γ = 1 - TPR
    delta_actual.append(delta)

    # Acceptance masks
    acc_ind = (ind_unc <= thr)
    acc_ood = (ood_unc <= thr)

    # Build accepted pool (enforce β=0 by setting OOD labels to -1)
    p_ind = ind_prob[acc_ind]
    l_ind = ind_label[acc_ind]
    p_ood = ood_prob[acc_ood]
    l_ood = np.full(len(p_ood), -1, dtype=int)  # Never covered (β=0)

    all_p = np.vstack([p_ind, p_ood])
    all_l = np.concatenate([l_ind, l_ood])

    # Empirical coverage
    sets = predict_set(all_p, alpha)
    covered = [all_l[j] in sets[j] for j in range(len(sets))]
    emp_cov = np.mean(covered)
    empirical_gaps_a.append(abs((1 - alpha) - emp_cov))

    # Theoretical worst-case coverage
    theo_cov = theoretical_coverage_worst_case(delta, gamma, pi_ood, alpha)
    theoretical_gaps_a.append(abs((1 - alpha) - theo_cov))

axs[0].plot(delta_actual, empirical_gaps_a, marker='x', linestyle='-',
            linewidth=2, color='blue', label='Empirical')
axs[0].plot(delta_actual, theoretical_gaps_a, marker='o', linestyle='--',
            linewidth=2, color='red', label='Theoretical (β=0)')
axs[0].set_xlabel(r'(a) FRR $\delta$')
axs[0].set_ylabel('Coverage Gap')
axs[0].grid(True)

# ------------------------------------------------------------
# Panel (b): Coverage gap vs. OOD proportion π
# ------------------------------------------------------------
pi_values = np.linspace(0.1, 0.9, 9)
fixed_delta = 0.2
i_fixed = np.argmin(np.abs(fpr_roc - fixed_delta))
fixed_thr = thresholds[i_fixed]
fixed_gamma = 1 - tpr_roc[i_fixed]

# Precompute accepted pools at fixed threshold
mask_ind_fixed = ind_unc <= fixed_thr
mask_ood_fixed = ood_unc <= fixed_thr

p_ind_fixed = ind_prob[mask_ind_fixed]
l_ind_fixed = ind_label[mask_ind_fixed]
p_ood_fixed = ood_prob[mask_ood_fixed]
l_ood_fixed = np.full(len(p_ood_fixed), -1, dtype=int)

empirical_gaps_b = []
theoretical_gaps_b = []

for pi_val in pi_values:
    # Theoretical
    theo_cov = theoretical_coverage_worst_case(fixed_delta, fixed_gamma, pi_val, alpha)
    theoretical_gaps_b.append(abs((1 - alpha) - theo_cov))

    # Subsample to achieve mixture π
    n_ind = int(len(p_ind_fixed) * (1 - pi_val))
    n_ood = int(len(p_ood_fixed) * pi_val)

    p_mix = np.vstack([p_ind_fixed[:n_ind], p_ood_fixed[:n_ood]])
    l_mix = np.concatenate([l_ind_fixed[:n_ind], l_ood_fixed[:n_ood]])

    sets = predict_set(p_mix, alpha)
    covered = [l_mix[j] in sets[j] for j in range(len(sets))]
    emp_cov = np.mean(covered)
    empirical_gaps_b.append(abs((1 - alpha) - emp_cov))

axs[1].plot(pi_values, empirical_gaps_b, marker='x', linestyle='-',
            linewidth=2, color='blue')
axs[1].plot(pi_values, theoretical_gaps_b, marker='o', linestyle='--',
            linewidth=2, color='red')
axs[1].set_xlabel(r'(b) OOD Proportion $\pi$')
axs[1].grid(True)
# ------------------------------------------------------------
# Panel (c): Coverage gap vs. miscoverage α (CORRECTED)
# ------------------------------------------------------------
alpha_values = np.linspace(0.01, 0.15, 10)
empirical_gaps_c = []
theoretical_gaps_c = []

# Use the same fixed threshold as panel (b) to maintain consistent δ and γ
# This gives us accepted InD and OOD pools
p_ind_accepted = p_ind_fixed  # Already computed in panel (b)
l_ind_accepted = l_ind_fixed
p_ood_accepted = p_ood_fixed
l_ood_accepted = l_ood_fixed  # Already set to -1 (β=0)

# Split accepted InD into calibration and test portions
# Calibration: used to compute CP quantile
# Test: will be mixed with OOD for evaluation
cal_ratio = 0.5  # Use 50% for calibration, 50% for test
n_cal = int(len(p_ind_accepted) * cal_ratio)

cal_p = p_ind_accepted[:n_cal]
cal_l = l_ind_accepted[:n_cal]
test_p_ind = p_ind_accepted[n_cal:]
test_l_ind = l_ind_accepted[n_cal:]

# Prepare OOD test samples
test_p_ood = p_ood_accepted
test_l_ood = l_ood_accepted  # Already -1 (enforces β=0)

# Create mixed test set maintaining proportion π
# Calculate how many OOD samples to mix with InD test samples
n_test_ind = len(test_p_ind)
n_test_ood = int(n_test_ind * pi_ood / (1 - pi_ood))  # Maintain π proportion
n_test_ood = min(n_test_ood, len(test_p_ood))  # Don't exceed available OOD

# Create the mixed test pool (fixed across all α values)
test_p_mixed = np.vstack([test_p_ind, test_p_ood[:n_test_ood]])
test_l_mixed = np.concatenate([test_l_ind, test_l_ood[:n_test_ood]])

print(f"\nPanel (c) setup:")
print(f"  Calibration set size: {len(cal_p)}")
print(f"  Test InD samples: {len(test_p_ind)}")
print(f"  Test OOD samples: {n_test_ood}")
print(f"  Empirical π in test: {n_test_ood / (n_test_ind + n_test_ood):.3f}")
print(f"  Target π: {pi_ood:.3f}\n")

for a in alpha_values:
    # Theoretical worst-case coverage at fixed detector performance
    theo_cov = theoretical_coverage_worst_case(fixed_delta, fixed_gamma, pi_ood, a)
    theo_gap = abs((1 - a) - theo_cov)
    theoretical_gaps_c.append(theo_gap)

    # Empirical: Calibrate CP on clean InD, test on contaminated mixture
    # Step 1: Compute nonconformity scores on calibration set
    scores = 1 - np.array([p[y] for p, y in zip(cal_p, cal_l)])

    # Step 2: Compute quantile threshold
    q = np.quantile(scores, 1 - a, method="higher")

    # Step 3: Apply to mixed test set (InD + OOD)
    # Prediction set includes all classes where prob >= 1 - q
    sets = [np.where(p >= 1 - q)[0] for p in test_p_mixed]

    # Step 4: Check coverage (OOD labels are -1, so never covered)
    covered = [test_l_mixed[j] in sets[j] for j in range(len(test_l_mixed))]
    emp_cov = np.mean(covered)
    emp_gap = abs((1 - a) - emp_cov)
    empirical_gaps_c.append(emp_gap)

    # Detailed output for debugging
    print(f"α={a:.3f}: target_cov={(1 - a):.4f}, "
          f"emp_cov={emp_cov:.4f}, theo_cov={theo_cov:.4f}, "
          f"emp_gap={emp_gap:.4f}, theo_gap={theo_gap:.4f}")

axs[2].plot(alpha_values, empirical_gaps_c, marker='x', linestyle='-',
            linewidth=2, color='blue')
axs[2].plot(alpha_values, theoretical_gaps_c, marker='o', linestyle='--',
            linewidth=2, color='red')
axs[2].set_xlabel(r'(c) Miscoverage $\alpha$')
axs[2].set_ylabel('Coverage Gap')
axs[2].set_ylim(-0.05, 0.20)  # Adjusted y-axis for new range
axs[2].grid(True)


# # ------------------------------------------------------------
# Panel (d): Coverage gap vs. FAR (γ = 1-TPR)
# ------------------------------------------------------------
gamma_targets = np.linspace(0.0, 0.9, 10)
empirical_gaps_d = []
theoretical_gaps_d = []
gamma_actual = []

for target_gamma in gamma_targets:
    # γ = 1 - TPR, so target TPR = 1 - target_gamma
    target_tpr = 1 - target_gamma
    i = np.argmin(np.abs(tpr_roc - target_tpr))
    thr = thresholds[i]
    delta = fpr_roc[i]
    gamma = 1 - tpr_roc[i]
    gamma_actual.append(gamma)

    # Theoretical
    theo_cov = theoretical_coverage_worst_case(delta, gamma, pi_ood, alpha)
    theoretical_gaps_d.append(abs((1 - alpha) - theo_cov))

    # Empirical
    acc_ind = ind_unc <= thr
    acc_ood = ood_unc <= thr

    p_i = ind_prob[acc_ind]
    l_i = ind_label[acc_ind]
    p_o = ood_prob[acc_ood]
    l_o = np.full(len(p_o), -1, dtype=int)

    # Subsample to original π
    n_i = int(len(p_i) * (1 - pi_ood))
    n_o = int(len(p_o) * pi_ood)

    p_mix = np.vstack([p_i[:n_i], p_o[:n_o]])
    l_mix = np.concatenate([l_i[:n_i], l_o[:n_o]])

    sets = predict_set(p_mix, alpha)
    covered = [l_mix[j] in sets[j] for j in range(len(l_mix))]
    emp_cov = np.mean(covered)
    empirical_gaps_d.append(abs((1 - alpha) - emp_cov))

axs[3].plot(gamma_actual, empirical_gaps_d, marker='x', linestyle='-',
            linewidth=2, color='blue')
axs[3].plot(gamma_actual, theoretical_gaps_d, marker='o', linestyle='--',
            linewidth=2, color='red')
axs[3].set_xlabel(r'(d) FAR $\gamma$')
axs[3].grid(True)

# ------------------------------------------------------------
# 5. Final touches
# ------------------------------------------------------------
fig.legend(
    ['Empirical', r'Theoretical ($\beta=0$)'],
    loc='upper center',
    ncol=2,
    fontsize=18,
    frameon=False
)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig('ood_worst_case_validation.pdf')
plt.show()