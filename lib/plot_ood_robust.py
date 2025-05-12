# import seaborn as sns
# sns.set(style="white", font_scale=1.5)
# import pickle
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import roc_curve
# from sklearn.model_selection import train_test_split
#
# # Load your data
# with open("../ood_leakage.pkl", "rb") as f:
#     data = pickle.load(f)
#
# # Extract relevant data
# iid_prob = data["iid"]["predictive_probs"]
# iid_label = data["iid"]["predictive_labels"]
# iid_unc = data["iid"]["predictive_uncertainty"]
#
# ood_prob = data["ood"]["predictive_probs"]
# ood_unc = data["ood"]["predictive_uncertainty"]
#
# # Calculate ROC curve for the detector
# #### 0 = IID (negative class)
# #### 1 = OOD (positive class)
# y_true = np.concatenate([np.zeros_like(iid_unc), np.ones_like(ood_unc)])
# y_score = np.concatenate([iid_unc, ood_unc])
#
# # fpr: false positive rate (fraction of IID misclassified as OOD)
# # fpr: Fraction of IID samples (label 0) mistakenly detected as OOD at each threshold
#
# # tpr: true positive rate (fraction of OOD correctly detected)
# # tpr: Fraction of OOD samples (label 1) correctly detected as OOD at each threshold
#
# # 1 − TPR = fraction of OOD samples mistakenly accepted as InD
#
# # thresholds: uncertainty cutoffs at which those rates are computed
# fpr, tpr, thresholds = roc_curve(y_true, y_score)
#
# pi = len(iid_unc) / (len(iid_unc) + len(ood_unc))
# alpha = 0.01
# 
# # Helper function to get prediction set
# def predict_set(probs, alpha):
#     sorted_indices = np.argsort(-probs, axis=1)
#     sorted_probs = np.take_along_axis(probs, sorted_indices, axis=1)
#     cumsum_probs = np.cumsum(sorted_probs, axis=1)
#     cutoff = (cumsum_probs < (1 - alpha)).sum(axis=1)
#     return [sorted_indices[i, :cutoff[i] + 1] for i in range(len(probs))]
#
# # Theoretical coverage
# def calculate_coverage(fpr_val, tpr_val, pi_val, alpha_val):
#     numerator = (1 - fpr_val) * (1 - pi_val) * (1 - alpha_val)
#     denominator = (1 - fpr_val) * (1 - pi_val) + pi_val * (1 - tpr_val)
#     return numerator / denominator
#
# fig, axs = plt.subplots(1, 4, figsize=(20, 5))
#
# # -------------------------- plot 1 --------------------------
# # Coverage gap vs FAR
# fpr_range = np.linspace(0.0, 0.9, 10)
# practical_gaps, theoretical_gaps, deltas  = [], [], []
#
# # # We vary FPR from 0-0.9 and compute the gap
# for target_fpr in fpr_range:
#
#     idx = np.argmin(np.abs(fpr - target_fpr))
#     # find the uncertainty threshold corresponding to the desired FPR
#     threshold = thresholds[idx]
#
#     # IID samples accepted at this threshold as InD
#     iid_accept_mask = iid_unc <= threshold
#     iid_reject_mask = iid_unc > threshold
#     delta = np.mean(iid_reject_mask)
#
#     deltas.append(delta)
#     # OOD samples mistakenly accepted as InD
#     ood_accept_mask = ood_unc <= threshold
#     print(f"Accepted InD: {np.sum(iid_accept_mask)}, Accepted OOD: {np.sum(ood_accept_mask)}")
#
#     iid_probs = iid_prob[iid_accept_mask]
#     iid_labels = iid_label[iid_accept_mask]
#     ood_probs = ood_prob[ood_accept_mask]
#     ood_labels = np.random.randint(0, iid_prob.shape[1], size=ood_probs.shape[0])
#
#     all_probs = np.concatenate([iid_probs, ood_probs])
#     all_labels = np.concatenate([iid_labels, ood_labels])
#     pred_sets = predict_set(all_probs, alpha)
#     covered = [all_labels[i] in pred_sets[i] for i in range(len(pred_sets))]
#     practical_coverage = np.mean(covered)
#     practical_gaps.append(  (1 - alpha) - practical_coverage)
#     theoretical_coverage = calculate_coverage(target_fpr, tpr[idx], pi, alpha)
#     theoretical_gaps.append(  (1 - alpha) - theoretical_coverage)
#
#
# axs[0].plot(deltas, practical_gaps, marker='x', linestyle='-', linewidth=2, color='blue')
# axs[0].plot(deltas, theoretical_gaps, marker='o', linestyle='--', linewidth=2, color='red')
#
# # axs[0].plot(fpr_range, practical_gaps, marker='x', linestyle='-', linewidth=2, color='blue')
# # axs[0].plot(fpr_range, theoretical_gaps, marker='o', linestyle='--', linewidth=2, color='red')
# axs[0].set_xlabel(r'a. FRR $\delta$')
# axs[0].set_ylabel('Coverage Gap')
# axs[0].grid(True)
#
#
# # -------------------------- plot 2 --------------------------
# # Coverage gap vs π
# pi_range = np.linspace(0.1, 0.9, 9)
# fixed_fpr = 0.2
# fixed_alpha = 0.01
# idx_fixed = np.argmin(np.abs(fpr - fixed_fpr))
# tpr_fixed_fpr = tpr[idx_fixed]
# thresh_fixed_fpr = thresholds[idx_fixed]
#
# practical_gaps_pi, theoretical_gaps_pi = [], []
#
# for pi_val in pi_range:
#     theoretical_coverage = calculate_coverage(fixed_fpr, tpr_fixed_fpr, pi_val, fixed_alpha)
#     theoretical_gaps_pi.append( abs ( (1 - fixed_alpha) - theoretical_coverage) )
#
#     iid_accept_mask = iid_unc <= thresh_fixed_fpr
#     ood_accept_mask = ood_unc <= thresh_fixed_fpr
#
#     iid_probs = iid_prob[iid_accept_mask]
#     iid_labels = iid_label[iid_accept_mask]
#
#     ood_probs = ood_prob[ood_accept_mask]
#     ood_labels = np.random.randint(0, iid_prob.shape[1], size=ood_probs.shape[0])
#
#     num_iid = int(len(iid_probs) * (1 - pi_val))
#     num_ood = int(len(ood_probs) * pi_val)
#
#     all_probs = np.concatenate([iid_probs[:num_iid], ood_probs[:num_ood]])
#     all_labels = np.concatenate([iid_labels[:num_iid], ood_labels[:num_ood]])
#
#     pred_sets = predict_set(all_probs, fixed_alpha)
#     covered = [all_labels[i] in pred_sets[i] for i in range(len(pred_sets))]
#     practical_coverage = np.mean(covered)
#     practical_gaps_pi.append( abs( (1 - fixed_alpha) - practical_coverage) )
#
# axs[1].plot(pi_range, practical_gaps_pi, marker='x', linestyle='-', linewidth=2, color='blue')
# axs[1].plot(pi_range, theoretical_gaps_pi, marker='o', linestyle='--', linewidth=2, color='red')
# axs[1].set_xlabel('b. OOD Proportion (π)')
# axs[1].grid(True)
#
# # # -------------------------- plot 3 --------------------------
# alpha_range = np.linspace(0.01, 0.20, 10)
# practical_gaps_alpha, theoretical_gaps_alpha = [], []
#
# # Accept only InD samples under OOD detector
# iid_accept_mask = iid_unc <= thresh_fixed_fpr
# iid_probs_accepted = iid_prob[iid_accept_mask]
# iid_labels_accepted = iid_label[iid_accept_mask]
#
# # Split into calibration and test sets
# cal_probs, test_probs, cal_labels, test_labels = train_test_split(
#     iid_probs_accepted, iid_labels_accepted, test_size=0.2, random_state=42
# )
#
# # --- Loop over α values ---
# for alpha_val in alpha_range:
#     # Step 1: Theoretical coverage gap
#     theoretical_coverage = calculate_coverage(fixed_fpr, tpr_fixed_fpr, pi, alpha_val)
#     theoretical_gaps_alpha.append( abs( (1 - alpha_val) - theoretical_coverage) )
#     theoretical_gaps_alpha = [abs(x) for x in theoretical_gaps_alpha]
#
#     # Step 2: Compute calibration scores: 1 - prob(true_label)
#     cal_scores = 1 - np.array([p[y] for p, y in zip(cal_probs, cal_labels)])
#
#     # Step 3: Determine threshold (quantile)
#     q = np.quantile(cal_scores, 1 - alpha_val, method="higher")
#
#     # Step 4: Predict sets for test samples: include labels with prob >= 1 - q
#     pred_sets = [np.where(p >= 1 - q)[0] for p in test_probs]
#
#     # Step 5: Evaluate empirical coverage on test set
#     covered = [test_labels[i] in pred_sets[i] for i in range(len(test_labels))]
#     practical_coverage = np.mean(covered)
#     practical_gaps_alpha.append(  (1 - alpha_val) - practical_coverage)
#
#     print(f"α={alpha_val:.2f}, practical={practical_coverage:.4f}, theoretical={theoretical_coverage:.4f}")
#
# axs[2].plot(alpha_range, practical_gaps_alpha, marker='x', linestyle='-', linewidth=2, color='blue')
# axs[2].plot(alpha_range, theoretical_gaps_alpha, marker='o', linestyle='--', linewidth=2, color='red')
# axs[2].set_xlabel('c. Alpha (α)')
# axs[2].set_ylim(-0.1, 0.15)
# axs[2].grid(True)
#
#
# # # -------------------------- plot 4 --------------------------
#
# fixed_alpha = 0.01
# fnr_range = np.linspace(0.0, 0.9, 10)
# # Locate threshold for each FNR value (i.e., 1 - TPR)
# practical_gaps_fnr = []
# theoretical_gaps_fnr = []
#
# for fnr_target in fnr_range:
#     # find index where tpr ≈ 1 - fnr
#     idx = np.argmin(np.abs((1-tpr) - fnr_target))
#     threshold = thresholds[idx]
#     fpr_val = fpr[idx]
#     tpr_val = tpr[idx]
#
#     # Theoretical gap
#     theoretical_coverage = calculate_coverage(fpr_val, tpr_val, pi, fixed_alpha)
#     theoretical_gaps_fnr.append(abs((1 - fixed_alpha) - theoretical_coverage))
#
#     # Practical gap
#     iid_accept_mask = iid_unc <= threshold
#     ood_accept_mask = ood_unc <= threshold
#     iid_probs = iid_prob[iid_accept_mask]
#     iid_labels = iid_label[iid_accept_mask]
#
#     ood_probs = ood_prob[ood_accept_mask]
#     ood_labels = np.random.randint(0, iid_prob.shape[1], size=len(ood_probs))
#
#     # # Mark OOD samples as invalid (-1)
#     # ood_labels = np.full(len(ood_probs), -1)
#
#     #
#     num_iid = int( len(iid_probs) * (pi) )
#     num_ood = int( len(ood_probs) * (1-pi) )
#
#     all_probs = np.concatenate([iid_probs[:num_iid], ood_probs[:num_ood]])
#     all_labels = np.concatenate([iid_labels[:num_iid], ood_labels[:num_ood]])
#     pred_sets = predict_set(all_probs, fixed_alpha)
#     covered = [all_labels[i] in pred_sets[i] for i in range(len(pred_sets))]
#
#     covered = [
#         all_labels[i] in pred_sets[i] if all_labels[i] != -1 else False
#         for i in range(len(pred_sets))
#     ]
#
#     practical_coverage = np.mean(covered)
#     practical_gaps_fnr.append(abs((1 - fixed_alpha) - practical_coverage))
#
# axs[3].plot(fnr_range, practical_gaps_fnr, marker='x', linestyle='-', linewidth=2, color='blue')
# axs[3].plot(fnr_range, theoretical_gaps_fnr, marker='o', linestyle='--', linewidth=2, color='red')
# axs[3].set_xlabel(r'd. FAR $\gamma$')
# axs[3].grid(True)
#
# # -------------------------- Global Legend --------------------------
# fig.legend(['Empirical Coverage Gap', 'Theoretical Coverage Gap'],
#            loc='upper center',
#            ncol=2,
#            fontsize=18,
#            frameon=False)
#
# # -------------------------- Save and Show --------------------------
# plt.tight_layout()
# plt.subplots_adjust(top=0.85)
# plt.savefig('ood_robust_full_comparison.pdf')
# plt.show()



import seaborn as sns
sns.set(style="white", font_scale=1.5)

import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

# Reproducibility
np.random.seed(0)

# --------------------------
# 1. Load and sanity-check data
# --------------------------
with open("../ood_leakage.pkl", "rb") as f:
    data = pickle.load(f)

iid_prob  = data["iid"]["predictive_probs"]
iid_label = data["iid"]["predictive_labels"]
iid_unc   = data["iid"]["predictive_uncertainty"]

ood_prob = data["ood"]["predictive_probs"]
ood_unc  = data["ood"]["predictive_uncertainty"]

# Ensure the lengths match
assert iid_unc.shape[0]   == iid_prob.shape[0]   == iid_label.shape[0], \
       f"IID length mismatch: unc={iid_unc.shape[0]}, prob={iid_prob.shape[0]}, lab={iid_label.shape[0]}"
assert ood_unc.shape[0]   == ood_prob.shape[0], \
       f"OOD length mismatch: unc={ood_unc.shape[0]}, prob={ood_prob.shape[0]}"

# --------------------------
# 2. Compute ROC curve for OOD detector
# --------------------------
y_true  = np.concatenate([np.zeros_like(iid_unc), np.ones_like(ood_unc)])
y_score = np.concatenate([iid_unc, ood_unc])
fpr, tpr, thresholds = roc_curve(y_true, y_score)

pi    = len(iid_unc) / (len(iid_unc) + len(ood_unc))  # P(IID)
alpha = 0.01                                          # miscoverage level

# --------------------------
# 3. Conformal helper functions
# --------------------------
def predict_set(probs: np.ndarray, alpha: float):
    """Return the conformal prediction sets for each row of `probs`."""
    # sort probabilities descending
    idx     = np.argsort(-probs, axis=1)
    sorted_ = np.take_along_axis(probs, idx, axis=1)
    cumsums = np.cumsum(sorted_, axis=1)
    # include all labels until cumsum < 1 - alpha
    cutoffs = (cumsums < (1 - alpha)).sum(axis=1)
    return [idx[i, : cutoffs[i] + 1] for i in range(len(probs))]

def calculate_coverage(fpr_val: float, tpr_val: float, pi_val: float, alpha_val: float):
    """
    Theoretical coverage among the accepted pool:
      P(accepted IID) = pi*(1 - FPR)
      P(accepted OOD) = (1 - pi)*(1 - TPR)
    Only IID points get the (1-alpha) guarantee.
    """
    num_iid = pi_val * (1 - fpr_val)
    num_ood = (1 - pi_val) * (1 - tpr_val)
    return (num_iid * (1 - alpha_val)) / (num_iid + num_ood)

# --------------------------
# 4. Prepare plotting canvas
# --------------------------
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# ------------------------------------------------------------
# Panel (a): Coverage gap vs. FRR (equivalently FPR on IID)
# ------------------------------------------------------------
fpr_targets       = np.linspace(0.0, 0.9, 10)
practical_gaps_1  = []
theoretical_gaps_1 = []
deltas           = []

for target_fpr in fpr_targets:
    # find closest FPR in ROC
    i = np.argmin(np.abs(fpr - target_fpr))
    thr = thresholds[i]

    # FRR = fraction of IID rejected
    reject_iid = (iid_unc > thr)
    delta = reject_iid.mean()
    deltas.append(delta)

    # Accepted masks
    acc_iid = ~reject_iid
    acc_ood = (ood_unc <= thr)

    # build the empirical accepted pool
    p_iid = iid_prob[acc_iid]
    l_iid = iid_label[acc_iid]
    p_ood = ood_prob[acc_ood]
    l_ood = np.full(len(p_ood), -1, dtype=int)  # never covered

    all_p = np.vstack([p_iid, p_ood])
    all_l = np.concatenate([l_iid, l_ood])

    # empirical coverage
    sets = predict_set(all_p, alpha)
    covered = [all_l[j] in sets[j] for j in range(len(sets))]
    emp_cov = np.mean(covered)
    practical_gaps_1.append(abs((1 - alpha) - emp_cov))

    # theoretical coverage
    theo_cov = calculate_coverage(fpr[i], tpr[i], pi, alpha)
    theoretical_gaps_1.append(abs((1 - alpha) - theo_cov))

axs[0].plot(deltas, practical_gaps_1, marker='x', linestyle='-', linewidth=2, color='blue')
axs[0].plot(deltas, theoretical_gaps_1, marker='o', linestyle='--', linewidth=2, color='red')
axs[0].set_xlabel(r'a. FRR $\delta$')
axs[0].set_ylabel('Coverage Gap')
axs[0].grid(True)

# ------------------------------------------------------------
# Panel (b): Coverage gap vs. mixture π at fixed FPR
# ------------------------------------------------------------
pi_values          = np.linspace(0.1, 0.9, 9)
fixed_fpr          = 0.2
i_fixed            = np.argmin(np.abs(fpr - fixed_fpr))
fixed_thr         = thresholds[i_fixed]
fixed_tpr         = tpr[i_fixed]

# precompute accepted pools at fixed threshold
mask_iid_fixed = iid_unc <= fixed_thr
mask_ood_fixed = ood_unc <= fixed_thr

p_iid_fixed = iid_prob[mask_iid_fixed]
l_iid_fixed = iid_label[mask_iid_fixed]
p_ood_fixed = ood_prob[mask_ood_fixed]
l_ood_fixed = np.full(len(p_ood_fixed), -1, dtype=int)

practical_gaps_2   = []
theoretical_gaps_2 = []

for pi_val in pi_values:
    # theoretical
    theo_cov = calculate_coverage(fixed_fpr, fixed_tpr, pi_val, alpha)
    theoretical_gaps_2.append(abs((1 - alpha) - theo_cov))

    # subsample to achieve mixture pi_val
    n_iid = int(len(p_iid_fixed) * pi_val)
    n_ood = int(len(p_ood_fixed) * (1 - pi_val))

    p_mix = np.vstack([p_iid_fixed[:n_iid], p_ood_fixed[:n_ood]])
    l_mix = np.concatenate([l_iid_fixed[:n_iid], l_ood_fixed[:n_ood]])

    sets = predict_set(p_mix, alpha)
    covered = [l_mix[j] in sets[j] for j in range(len(sets))]
    emp_cov = np.mean(covered)
    practical_gaps_2.append(abs((1 - alpha) - emp_cov))

axs[1].plot(pi_values, practical_gaps_2, marker='x', linestyle='-', linewidth=2, color='blue')
axs[1].plot(pi_values, theoretical_gaps_2, marker='o', linestyle='--', linewidth=2, color='red')
axs[1].set_xlabel('b. OOD Proportion (π)')
axs[1].grid(True)

# ------------------------------------------------------------
# Panel (c): Coverage gap vs. significance α on IID hold-out
# ------------------------------------------------------------
alpha_values       = np.linspace(0.01, 0.20, 10)
practical_gaps_3   = []
theoretical_gaps_3 = []

# split accepted IID into calibration / test
p_acc_iid = p_iid_fixed
l_acc_iid = l_iid_fixed
cal_p, test_p, cal_l, test_l = train_test_split(
    p_acc_iid, l_acc_iid, test_size=0.2, random_state=42
)

for a in alpha_values:
    # theoretical at fixed detector performance
    theo_cov = calculate_coverage(fixed_fpr, fixed_tpr, pi, a)
    theoretical_gaps_3.append(abs((1 - a) - theo_cov))

    # conformal calibration on IID
    scores = 1 - np.array([p[y] for p, y in zip(cal_p, cal_l)])
    q = np.quantile(scores, 1 - a, method="higher")
    sets = [np.where(p >= 1 - q)[0] for p in test_p]
    covered = [test_l[j] in sets[j] for j in range(len(test_l))]
    emp_cov = np.mean(covered)
    practical_gaps_3.append(abs((1 - a) - emp_cov))

    print(f"α={a:.2f}: empirical={emp_cov:.4f}, theoretical={theo_cov:.4f}")

axs[2].plot(alpha_values, practical_gaps_3, marker='x', linestyle='-', linewidth=2, color='blue')
axs[2].plot(alpha_values, theoretical_gaps_3, marker='o', linestyle='--', linewidth=2, color='red')
axs[2].set_xlabel('c. Significance level (α)')
axs[2].set_ylim(-0.1, 0.15)
axs[2].grid(True)

# ------------------------------------------------------------
# Panel (d): Coverage gap vs. FNR (1 − TPR)
# ------------------------------------------------------------
fnr_targets         = np.linspace(0.0, 0.9, 10)
practical_gaps_4    = []
theoretical_gaps_4  = []

for fnr in fnr_targets:
    # match 1 - TPR
    i = np.argmin(np.abs((1 - tpr) - fnr))
    thr = thresholds[i]
    fpr_val = fpr[i]
    tpr_val = tpr[i]

    # theoretical
    theo_cov = calculate_coverage(fpr_val, tpr_val, pi, alpha)
    theoretical_gaps_4.append(abs((1 - alpha) - theo_cov))

    # empirical: accept mask
    acc_iid = iid_unc <= thr
    acc_ood = ood_unc <= thr

    p_i = iid_prob[acc_iid]
    l_i = iid_label[acc_iid]
    p_o = ood_prob[acc_ood]
    l_o = np.full(len(p_o), -1, dtype=int)

    # subsample to original π
    n_i = int(len(p_i) * pi)
    n_o = int(len(p_o) * (1 - pi))

    p_mix = np.vstack([p_i[:n_i], p_o[:n_o]])
    l_mix = np.concatenate([l_i[:n_i], l_o[:n_o]])

    sets = predict_set(p_mix, alpha)
    covered = [l_mix[j] in sets[j] for j in range(len(l_mix))]
    emp_cov = np.mean(covered)
    practical_gaps_4.append(abs((1 - alpha) - emp_cov))

axs[3].plot(fnr_targets, practical_gaps_4, marker='x', linestyle='-', linewidth=2, color='blue')
axs[3].plot(fnr_targets, theoretical_gaps_4, marker='o', linestyle='--', linewidth=2, color='red')
axs[3].set_xlabel(r'd. OOD acceptance (FNR)')
axs[3].grid(True)

# ------------------------------------------------------------
# 5. Final touches
# ------------------------------------------------------------
fig.legend(
    ['Empirical Coverage Gap', 'Theoretical Coverage Gap'],
    loc='upper center',
    ncol=2,
    fontsize=18,
    frameon=False
)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig('ood_robust_full_comparison.pdf')
plt.show()

