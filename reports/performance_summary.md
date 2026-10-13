# Performance summary

Source: `notebooks/10_performance_metrics.ipynb` (tables in `reports/performance_*.csv`). Every number is on the untouched test split of the model's own split protocol. Intervals are 95% cluster-bootstrap intervals over behavior groups (1,000 resamples, 500 buckets of whole groups), so look-alike flows are resampled together.

- **Group split**: test behaviors never seen in training (662,347 flows). The honest generalization estimate.
- **Random split**: stratified random split (719,661 flows). Comparable to published work.
- **Ceiling** (flow behavior alone, ports ignored): 0.808 macro-F1 / 0.809 accuracy for the 13 attack types, 0.966 / 0.966 for the 8 families.

## 13 attack types, group split

| Model | Macro-F1 [95% CI] | Accuracy [95% CI] | MCC | Macro precision | Macro recall |
|---|---|---|---|---|---|
| XGBoost | 0.783 [0.768, 0.796] | 0.802 [0.776, 0.826] | 0.784 | 0.796 | 0.781 |
| XGBoost (no ports) | 0.760 [0.741, 0.772] | 0.775 [0.745, 0.803] | 0.756 | 0.770 | 0.761 |
| CNN-BiLSTM-Attention v2 + offsets | 0.748 [0.723, 0.771] | 0.774 [0.736, 0.812] | 0.755 | 0.772 | 0.750 |
| CNN-BiLSTM-Attention v2 | 0.744 [0.720, 0.769] | 0.768 [0.729, 0.807] | 0.748 | 0.766 | 0.746 |
| CNN-BiLSTM-Attention v1 | 0.690 [0.665, 0.705] | 0.718 [0.685, 0.756] | 0.695 | 0.705 | 0.698 |
| Logistic regression | 0.686 [0.668, 0.698] | 0.697 [0.671, 0.726] | 0.671 | 0.693 | 0.689 |
| Base paper's CNN, 10 epochs | 0.686 [0.660, 0.702] | 0.716 [0.683, 0.753] | | | |
| Base paper's LSTM, 1 epoch | 0.634 [0.598, 0.660] | 0.664 [0.623, 0.700] | | | |
| Base paper's CNN, 1 epoch | 0.625 [0.595, 0.652] | 0.673 [0.630, 0.712] | | | |

## 13 attack types, random split

| Model | Macro-F1 [95% CI] | Accuracy [95% CI] |
|---|---|---|
| LightGBM | 0.794 [0.774, 0.808] | 0.790 [0.765, 0.818] |
| XGBoost | 0.791 [0.770, 0.805] | 0.787 [0.761, 0.815] |
| CNN-BiLSTM-Attention v2 | 0.770 [0.745, 0.789] | 0.767 [0.732, 0.799] |
| Base paper's CNN, 10 epochs | 0.676 [0.657, 0.693] | 0.718 [0.679, 0.754] |
| Base paper's CNN, 1 epoch | 0.648 [0.630, 0.664] | 0.694 [0.660, 0.733] |
| Base paper's LSTM, 1 epoch | 0.615 [0.584, 0.640] | 0.667 [0.622, 0.709] |

## 8 attack families

| Split | Model | Macro-F1 [95% CI] | Accuracy [95% CI] |
|---|---|---|---|
| group | XGBoost + class offsets (served) | 0.981 [0.976, 0.985] | 0.981 [0.977, 0.984] |
| group | XGBoost | 0.975 [0.967, 0.980] | 0.973 [0.968, 0.978] |
| random | XGBoost | 0.956 [0.926, 0.981] | 0.952 [0.910, 0.980] |

## Attack versus benign (any attack class counts as attack)

This part of the problem is solved. On the group split: XGBoost detects 99.998% of attack flows with a false positive rate under 0.001% (ROC AUC 1.000); the network v2 detects 99.99% with a false positive rate of 0.005% (5 benign flows in 100,000); the v1 network detected 99.89% with 0.04% false positives; logistic regression detects 99.82% with 0.67% false positives. Full table: `reports/performance_binary.csv`.

## What the intervals say

Paired bootstrap of macro-F1 differences on the group split (`reports/performance_paired_differences.csv`):

- **The v2 input fix is real**: v2 beats v1 by 0.058 [0.031, 0.086]. Before it, the network was statistically tied with logistic regression (difference 0.003 [-0.014, 0.019]).
- **XGBoost still beats the network**: 0.038 ahead of v2 [0.017, 0.061]. But the network is **statistically tied with XGBoost without ports** (difference 0.013 [-0.010, 0.037]), so the remaining gap is mostly the information that ports carry.
- **Ports matter**: XGBoost with ports beats XGBoost without them by 0.025 [0.017, 0.035].
- **Class offsets on the network are not significant**: +0.004 [-0.005, 0.017] (interval includes zero). On the family task they are: +0.007 [0.005, 0.009].
- **LightGBM vs XGBoost (random split)**: +0.003 [0.002, 0.005]: real but too small to matter.
- Intervals on the 13-class group split are about plus or minus 0.015, so differences smaller than that between two models should not be reported as improvements. This applies to the 0.792 macro-F1 of LightGBM with engineered features and offsets from the earlier experiments (`reports/push_group_f1_results.json`): those models were not saved and are not in this report, so that figure has no interval.

## The base paper's models on our 13-class task

The paper has no 13-class result, so its recipe was run on our task (`experiments/paper_models_13class.py`): its CNN and LSTM architectures, up to 50,000 training rows per class (650,000 rows), standard scaling, Adam at 1e-3, batch 64, 1 epoch as in the paper (10 epochs for the CNN as a check), scored on the full test split of each protocol.

- **Our network beats them clearly.** Group split, macro-F1 difference of v2 + offsets over the paper's CNN (1 epoch): 0.124 [0.092, 0.158]; over the paper's LSTM: 0.118 [0.083, 0.157]; over the paper's CNN trained for 10 epochs: 0.067 [0.041, 0.094].
- **XGBoost beats them by more**: 0.158 [0.129, 0.192] over the 1-epoch CNN, 0.101 [0.075, 0.127] over the 10-epoch CNN.
- **The paper's best model is only as good as our first network and logistic regression.** The 10-epoch CNN ties v1 (difference 0.005 [-0.000, 0.011]) and logistic regression (0.002 [-0.016, 0.020]). The v2 input fix is what moves a network past that.
- The paper's CNN and LSTM are not distinguishable at 1 epoch (difference -0.006 [-0.021, 0.009]).
- Attack versus benign is still easy for them (detection 99.8% to 99.99%, false positives 0.1% to 0.35%), so their weakness is telling the attack types apart.

## Where the 13-class errors are (group split, F1)

| Class | XGBoost | Network v2 + offsets |
|---|---|---|
| BENIGN, NTP, TFTP, MSSQL, Syn | 0.96 to 1.00 | 0.88 to 1.00 |
| UDPLag | 0.84 | 0.84 |
| NetBIOS | 0.78 | 0.79 |
| SNMP | 0.76 | 0.67 |
| DNS | 0.69 | 0.67 |
| SSDP | 0.60 | 0.61 |
| UDP | 0.61 | 0.52 |
| LDAP | 0.56 | 0.51 |
| Portmap | 0.42 | 0.29 |

DNS, LDAP, Portmap, SSDP and UDP are the weak classes for every model, the ones the ceiling analysis marks as behaviorally ambiguous.
