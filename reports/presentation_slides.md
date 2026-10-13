# Presentation slide content

Corrected against what the project actually does, with the final results. Numbers are on the group split (test behaviors never seen in training) unless stated. Sources are in `reports/`.

---

## Slide 1: Problem identification and literature survey

**Problem**
- DDoS attacks flood servers with traffic from many sources, causing service outages
- Signature-based detection fails on new and variant attacks

**Objective**
- Multi-class DDoS detection: identify the attack type, not just attack vs. normal
- Compare a CNN + BiLSTM + Attention network with tree-based baselines under a leakage-aware evaluation

**Literature gaps** (fill in your own reading list: the project's base paper is Fu et al. 2025, a survey)
- Weak generalization to unseen and zero-day attacks
- Privacy and robustness rarely combined, with high compute overhead
- Narrow validation: few attack types, near-100% accuracy claims that we could not reproduce (reported 99.55%, our reproduction about 85%)

---

## Slide 2: Handling datasets and data preprocessing

**Dataset:** CIC-DDoS2019 (about 30 GB, 18 CSVs from two capture days, 13 classes)

**Scale**
- Original: about 70.43 M rows
- After cleaning: about 69.94 M rows (99.3% retained)

**Challenges and fixes**
- Memory limits -> files processed one at a time, saved as Parquet
- Inf values -> converted to NaN; the NaNs are zero-duration flows, kept and imputed
- 482,799 duplicate rows removed; the WebDDoS class (439 rows) dropped as too rare
- Class imbalance (183x between largest and smallest) -> about 300k rows per class, balanced class weights
- Evaluation leakage -> two splits: stratified random and group-aware (identical flow behaviors never straddle train and test)
- Spoofable identifiers (IPs, Flow ID, timestamp) -> dropped

**Feature selection:** flow-based statistics only (rates, sizes, flags); 88 raw columns -> 60 features after removing identifiers, 12 constant and 9 redundant features

---

## Slide 3: Algorithmic design and approach

**Model:** CNN + BiLSTM + Attention, 13-class softmax
- CNN -> local patterns across neighboring features
- BiLSTM -> context across the feature sequence, forward and backward
- Attention -> weights the most informative steps before pooling
- Inputs: 60 flow features, log-scaled and standardized, fed as a sequence; the two ports go through an embedding layer

**Key finding about the model:** the first version trailed XGBoost by 0.09 macro-F1. The cause was input handling (no log transform on heavy-tailed features, ports scaled as magnitudes), not the architecture. Fixing both raised macro-F1 from 0.690 to 0.744.

**Baselines on identical splits:** XGBoost, LightGBM, Random Forest, Logistic Regression

**Explainability**
- TreeSHAP for XGBoost: global importance per class, and per-flow contributions in the app (Source Port, packet sizes and TCP flags lead)
- Attention weights for the network: after the fix it looks at ports and packet-length features and agrees more with SHAP (rank correlation 0.41, was 0.30)

**Deployment:** web application (FastAPI backend, React interface): classify a sample or an uploaded CSV, per-class metrics, confusion matrix, explanations, and a simulated traffic replay

*(Flow diagram: Input, 60 features + 2 ports -> Conv1D x2 -> BiLSTM -> Attention pooling -> (+ port embedding) -> Dense -> Attack class)*

---

## Slide 4: Results and key findings

**13 attack types (group split)**

| Model | Macro-F1 | Accuracy |
|---|---|---|
| Behavior-only ceiling | 0.808 | 0.809 |
| XGBoost | 0.783 | 0.802 |
| LightGBM + features + class offsets | 0.792 | 0.807 |
| CNN-BiLSTM-Attention v2 (with class offsets) | 0.748 | 0.774 |
| CNN-BiLSTM-Attention v1 | 0.690 | 0.718 |

- Ceiling: about 63% of flows are identical to a flow with a different label (for example DNS vs LDAP vs SNMP), so no model on flow features can exceed about 0.81
- XGBoost reaches about 97% of the ceiling; more rows or heavier tuning add at most 0.01 (learning curve: 25% of the rows already gives the same score)
- The base paper's CNN and LSTM, trained their way on our 13 classes: macro-F1 0.63 to 0.69 (1 epoch to 10 epochs), about 0.06 to 0.12 below our network and 0.10 to 0.16 below XGBoost (all differences significant)
- Averaging XGBoost with the network is worse than XGBoost alone (0.766), so the weaker model does not help

**8 attack families (merging the confusable types)**

| | Random split | Group split |
|---|---|---|
| XGBoost | 0.956 | 0.975 |
| With tuned class offsets | 0.970 | 0.981 |

- Family ceiling is 0.966; the ambiguity was always inside a family
- Class offsets: a constant per class added to the log-probabilities, fitted on a validation slice of the training data (never the test set)

**Base paper check**
- Reproduced the paper's setup: about 85% accuracy vs. 99.55% claimed
- The paper's number exceeds the no-port ceiling for that task (0.887), so it likely rests on an undisclosed methodological issue
