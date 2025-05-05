# import seaborn as sns
# sns.set(style="white", font_scale=1.2)
#
# import pickle
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import roc_curve
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
# y_true = np.concatenate([np.zeros_like(iid_unc), np.ones_like(ood_unc)])
# y_score = np.concatenate([iid_unc, ood_unc])
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
#
# def calculate_coverage(fpr_val, tpr_val, pi_val, alpha_val):
#     numerator = (1 - fpr_val) * (1 - pi_val) * (1 - alpha_val)
#     denominator = (1 - fpr_val) * (1 - pi_val) + pi_val * (1 - tpr_val)
#     return numerator / denominator
#
# plt.figure(figsize=(16, 5))
#
# # Coverage gap vs FAR
# fpr_range = np.linspace(0.0, 0.9, 10)
# practical_gaps, theoretical_gaps = [], []
#
# for target_fpr in fpr_range:
#     idx = np.argmin(np.abs(fpr - target_fpr))
#     threshold = thresholds[idx]
#
#     iid_accept_mask = iid_unc <= threshold
#     ood_accept_mask = ood_unc <= threshold
#
#     iid_probs = iid_prob[iid_accept_mask]
#     iid_labels = iid_label[iid_accept_mask]
#     ood_probs = ood_prob[ood_accept_mask]
#     ood_labels = np.random.randint(0, iid_prob.shape[1], size=ood_probs.shape[0])
#
#     all_probs = np.concatenate([iid_probs, ood_probs])
#     all_labels = np.concatenate([iid_labels, ood_labels])
#
#     pred_sets = predict_set(all_probs, alpha)
#     covered = [all_labels[i] in pred_sets[i] for i in range(len(pred_sets))]
#     practical_coverage = np.mean(covered)
#     practical_gaps.append((1 - alpha) - practical_coverage)
#
#     theoretical_coverage = calculate_coverage(target_fpr, tpr[idx], pi, alpha)
#     theoretical_gaps.append((1 - alpha) - theoretical_coverage)
#
# plt.subplot(1, 3, 1)
# plt.plot(fpr_range, practical_gaps, marker='x', linestyle='-', linewidth=2, color='blue', label='Practical Coverage Gap')
# plt.plot(fpr_range, theoretical_gaps, marker='o', linestyle='--', linewidth=2, color='red', label='Theoretical Coverage Gap')
# plt.title('Coverage Gap vs False Acceptance Rate (FAR)')
# plt.xlabel('False Acceptance Rate (FAR)')
# plt.ylabel('Coverage Gap')
# plt.grid(True)
# plt.legend()
#
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
#     theoretical_gaps_pi.append((1 - fixed_alpha) - theoretical_coverage)
#
#     iid_accept_mask = iid_unc <= thresh_fixed_fpr
#     ood_accept_mask = ood_unc <= thresh_fixed_fpr
#
#     iid_probs = iid_prob[iid_accept_mask]
#     iid_labels = iid_label[iid_accept_mask]
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
#     practical_gaps_pi.append((1 - fixed_alpha) - practical_coverage)
#
# plt.subplot(1, 3, 2)
# plt.plot(pi_range, practical_gaps_pi, marker='x', linestyle='-', linewidth=2, color='blue', label='Practical Coverage Gap')
# plt.plot(pi_range, theoretical_gaps_pi, marker='o', linestyle='--', linewidth=2, color='red', label='Theoretical Coverage Gap')
# plt.title(f'Coverage Gap vs OOD Proportion (FAR={fixed_fpr}, α={fixed_alpha})')
# plt.xlabel('OOD Proportion (π)')
# plt.ylabel('Coverage Gap')
# plt.grid(True)
# plt.legend()
#
# # Coverage gap vs α
# alpha_range = np.linspace(0.01, 0.20, 10)
# practical_gaps_alpha, theoretical_gaps_alpha = [], []
#
# iid_accept_mask = iid_unc <= thresh_fixed_fpr
# ood_accept_mask = ood_unc <= thresh_fixed_fpr
#
# iid_probs = iid_prob[iid_accept_mask]
# iid_labels = iid_label[iid_accept_mask]
# ood_probs = ood_prob[ood_accept_mask]
# ood_labels = np.random.randint(0, iid_prob.shape[1], size=ood_probs.shape[0])
#
# all_probs = np.concatenate([iid_probs, ood_probs])
# all_labels = np.concatenate([iid_labels, ood_labels])
#
# for alpha_val in alpha_range:
#     theoretical_coverage = calculate_coverage(fixed_fpr, tpr_fixed_fpr, pi, alpha_val)
#     theoretical_gaps_alpha.append((1 - alpha_val) - theoretical_coverage)
#
#     pred_sets = predict_set(all_probs, alpha_val)
#     covered = [all_labels[i] in pred_sets[i] for i in range(len(pred_sets))]
#     practical_coverage = np.mean(covered)
#     practical_gaps_alpha.append((1 - alpha_val) - practical_coverage)
#
#     print(f"α={alpha_val:.2f}, practical={practical_coverage:.4f}, theoretical={theoretical_coverage:.4f}")
#
# plt.subplot(1, 3, 3)
# plt.plot(alpha_range, practical_gaps_alpha, marker='x', linestyle='-', linewidth=2, color='blue', label='Practical Coverage Gap')
# plt.plot(alpha_range, theoretical_gaps_alpha, marker='o', linestyle='--', linewidth=2, color='red', label='Theoretical Coverage Gap')
# plt.title(f'Coverage Gap vs Alpha (FAR={fixed_fpr}, π={pi:.2f})')
# plt.xlabel('Alpha (α)')
# plt.ylabel('Coverage Gap')
# plt.grid(True)
# plt.legend()
#
# plt.tight_layout()
# plt.savefig('ood_robust_full_comparison.pdf')
# plt.show()

import seaborn as sns
sns.set(style="white", font_scale=1.2)

import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

# Load your data
with open("../ood_leakage.pkl", "rb") as f:
    data = pickle.load(f)

# Extract relevant data
iid_prob = data["iid"]["predictive_probs"]
iid_label = data["iid"]["predictive_labels"]
iid_unc = data["iid"]["predictive_uncertainty"]
ood_prob = data["ood"]["predictive_probs"]
ood_unc = data["ood"]["predictive_uncertainty"]

# ROC curve for uncertainty-based OOD detector
y_true = np.concatenate([np.zeros_like(iid_unc), np.ones_like(ood_unc)])
y_score = np.concatenate([iid_unc, ood_unc])
fpr, tpr, thresholds = roc_curve(y_true, y_score)

pi = len(iid_unc) / (len(iid_unc) + len(ood_unc))
alpha_range = np.linspace(0.01, 0.20, 10)
fixed_fpr = 0.2
idx_fixed = np.argmin(np.abs(fpr - fixed_fpr))
tpr_fixed_fpr = tpr[idx_fixed]
thresh_fixed_fpr = thresholds[idx_fixed]

# Helper function
def predict_set(probs, alpha):
    sorted_indices = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, sorted_indices, axis=1)
    cumsum_probs = np.cumsum(sorted_probs, axis=1)
    cutoff = (cumsum_probs < (1 - alpha)).sum(axis=1)
    return [sorted_indices[i, :cutoff[i] + 1] for i in range(len(probs))]

def calculate_coverage(fpr_val, tpr_val, pi_val, alpha_val):
    numerator = (1 - fpr_val) * (1 - pi_val) * (1 - alpha_val)
    denominator = (1 - fpr_val) * (1 - pi_val) + pi_val * (1 - tpr_val)
    return numerator / denominator

# Filter accepted samples
iid_accept_mask = iid_unc <= thresh_fixed_fpr
ood_accept_mask = ood_unc <= thresh_fixed_fpr

iid_probs = iid_prob[iid_accept_mask]
iid_labels = iid_label[iid_accept_mask]
ood_probs = ood_prob[ood_accept_mask]
ood_labels = np.random.randint(0, iid_prob.shape[1], size=ood_probs.shape[0])

all_probs = np.concatenate([iid_probs, ood_probs])
all_labels = np.concatenate([iid_labels, ood_labels])
is_ood = np.concatenate([np.zeros(len(iid_labels)), np.ones(len(ood_labels))])

# Gap tracking
practical_gaps_alpha, theoretical_gaps_alpha = [], []
id_coverage_list, ood_coverage_list = [], []

plt.figure(figsize=(12, 6))

for alpha_val in alpha_range:
    theoretical_coverage = calculate_coverage(fixed_fpr, tpr_fixed_fpr, pi, alpha_val)
    theoretical_gaps_alpha.append((1 - alpha_val) - theoretical_coverage)

    pred_sets = predict_set(all_probs, alpha_val)
    covered = [all_labels[i] in pred_sets[i] for i in range(len(pred_sets))]

    iid_cov = np.mean([covered[i] for i in range(len(covered)) if is_ood[i] == 0])
    ood_cov = np.mean([covered[i] for i in range(len(covered)) if is_ood[i] == 1])

    id_coverage_list.append(iid_cov)
    ood_coverage_list.append(ood_cov)

    weighted_cov = (len(iid_probs) * iid_cov + len(ood_probs) * ood_cov) / (len(iid_probs) + len(ood_probs))
    practical_gaps_alpha.append((1 - alpha_val) - weighted_cov)

    print(f"α={alpha_val:.2f}, InD={iid_cov:.4f}, OOD={ood_cov:.4f}, Practical={weighted_cov:.4f}, Theoretical={theoretical_coverage:.4f}")

# Plot practical and theoretical coverage gap vs alpha
plt.subplot(1, 2, 1)
plt.plot(alpha_range, practical_gaps_alpha, marker='x', linestyle='-', linewidth=2, label='Practical Coverage Gap')
plt.plot(alpha_range, theoretical_gaps_alpha, marker='o', linestyle='--', linewidth=2, label='Theoretical Coverage Gap')
plt.title('Coverage Gap vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Coverage Gap')
plt.grid(True)
plt.legend()

# Plot InD and OOD coverage curves
plt.subplot(1, 2, 2)
plt.plot(alpha_range, id_coverage_list, marker='o', label='InD Coverage')
plt.plot(alpha_range, ood_coverage_list, marker='x', label='OOD Coverage')
plt.title('Separate Coverage vs Alpha')
plt.xlabel('Alpha')
plt.ylabel('Coverage')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("ood_coverage_split_vs_alpha.pdf")
plt.show()


