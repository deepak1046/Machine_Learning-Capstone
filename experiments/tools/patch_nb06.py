import nbformat, json
p = 'notebooks/06_final_evaluation.ipynb'
nb = nbformat.read(p, as_version=4)
def sub(i, old, new):
    c = nb.cells[i]; assert old in c.source, (i, old[:70]); c.source = c.source.replace(old, new)

sub(2, 'unpooled = pd.read_csv(REPORTS / "hybrid_results.csv").assign(model="CNN-BiLSTM-Attention (un-pooled)")\n',
'''unpooled = pd.read_csv(REPORTS / "hybrid_results.csv").assign(model="CNN-BiLSTM-Attention (un-pooled)")
network_v2 = pd.read_csv(REPORTS / "hybrid_v2_results.csv").assign(model="CNN-BiLSTM-Attention (log + ports)")

# Tuned decision rules (experiments/push_group_f1.py and push_f1.py): per-class log-probability offsets fitted
# on a validation slice carved out of TRAIN, so the test set stays untouched.
push_group = json.load(open(REPORTS / "push_group_f1_results.json"))
push_random = pd.read_csv(REPORTS / "push_f1_13_random.csv").set_index("model")
tuned = pd.DataFrame([
    {"split": "group", "model": "LightGBM + features + offsets", "macro_f1": push_group["LGB-eng_offsets"][0], "accuracy": push_group["LGB-eng_offsets"][1]},
    {"split": "random", "model": "LightGBM + features + offsets", "macro_f1": push_random.loc["LightGBM + eng + tuned offsets", "macro_f1"], "accuracy": push_random.loc["LightGBM + eng + tuned offsets", "accuracy"]},
    {"split": "group", "model": "Ensemble (XGBoost, LightGBM) + offsets", "macro_f1": push_group["Ensemble_offsets"][0], "accuracy": push_group["Ensemble_offsets"][1]},
    {"split": "random", "model": "Ensemble (XGBoost, LightGBM) + offsets", "macro_f1": push_random.loc["Ensemble + tuned offsets", "macro_f1"], "accuracy": push_random.loc["Ensemble + tuned offsets", "accuracy"]},
])
net_offsets = pd.read_csv(REPORTS / "hybrid_v2_offsets_results.csv")
tuned = pd.concat([tuned, pd.DataFrame({
    "split": net_offsets["split"], "model": "CNN-BiLSTM-Attention (log + ports) + offsets",
    "macro_f1": net_offsets["test_macro_f1_offsets"], "accuracy": net_offsets["test_accuracy_offsets"]})], ignore_index=True)
''')
sub(2, '''long = pd.concat([base[cols + ["seconds"]], improve[cols], pooled[cols + ["seconds"]], unpooled[cols + ["seconds"]]],
                 ignore_index=True)''', '''long = pd.concat([base[cols + ["seconds"]], improve[cols], pooled[cols + ["seconds"]], unpooled[cols + ["seconds"]],
                  network_v2[cols + ["seconds"]], tuned[cols]], ignore_index=True)''')
sub(3, 'hybrids = ["CNN-BiLSTM-Attention (pooled)", "CNN-BiLSTM-Attention (un-pooled)"]',
       'hybrids = ["CNN-BiLSTM-Attention (pooled)", "CNN-BiLSTM-Attention (un-pooled)", "CNN-BiLSTM-Attention (log + ports)"]')
sub(3, 'label_rows=(order[0], "CNN-BiLSTM-Attention (un-pooled)"),', 'label_rows=(order[0], "CNN-BiLSTM-Attention (log + ports)"),')
sub(4, 'per_class_13 = pd.DataFrame({', '''hyb_v2 = pd.read_csv(REPORTS / "hybrid_v2_per_class_f1.csv", index_col=0)

per_class_13 = pd.DataFrame({''')
sub(4, '''    "Hybrid pooled | group": hyb_p["CNN-BiLSTM-Attention (group)"],
}).loc[class_names]''', '''    "Hybrid pooled | group": hyb_p["CNN-BiLSTM-Attention (group)"],
    "Hybrid v2 | random": hyb_v2.get("CNN-BiLSTM-Attention v2 (random)"),
    "Hybrid v2 | group": hyb_v2["CNN-BiLSTM-Attention v2 (group)"],
}).loc[class_names].dropna(axis=1, how="all")''')
sub(5, 'fam_summary.to_csv(REPORTS / "final_summary_family.csv")', '''# Tuned decision rule (class offsets fitted on a validation slice of train). The served family model is XGBoost + offsets.
offsets_served = pd.read_csv(REPORTS / "xgb_offsets_results.csv").set_index("task").loc["family"]
push_fam = {s: pd.read_csv(REPORTS / f"push_f1_family_{s}.csv").set_index("model").loc["Ensemble + tuned offsets"] for s in ("random", "group")}
tuned_rows = {
    "random (ensemble + offsets)": (push_fam["random"]["macro_f1"], push_fam["random"]["accuracy"]),
    "group (ensemble + offsets)": (push_fam["group"]["macro_f1"], push_fam["group"]["accuracy"]),
    "group (XGBoost + offsets, served)": (offsets_served["test_macro_f1_offsets"], offsets_served["test_accuracy_offsets"]),
}
extra = pd.DataFrame({
    name: {"macro-F1": f1, "accuracy": acc, "ceiling macro-F1": ceil_fam["macro_f1"], "ceiling accuracy": ceil_fam["accuracy"],
           "macro-F1 minus ceiling": f1 - ceil_fam["macro_f1"]}
    for name, (f1, acc) in tuned_rows.items()
}).T
fam_summary = pd.concat([fam_summary, extra])
fam_summary.index.name = "split"
fam_summary.to_csv(REPORTS / "final_summary_family.csv")''')
# headline numbers: insert before the line that prints the 8-family heading
c = nb.cells[7]
lines = c.source.split("\n")
at = next(i for i, l in enumerate(lines) if "8-FAMILY TASK" in l)
insert = '''hv = "CNN-BiLSTM-Attention (log + ports)"
for split in ("random", "group"):
    if hv in macro.index and not np.isnan(macro.loc[hv, split]):
        print(f"  network v2 {split}: {macro.loc[hv, split]:.3f} ({100 * macro.loc[hv, split] / ceil_13['macro_f1']:.1f}% of ceiling)"
              f"   gap to XGBoost {macro.loc[xg, split] - macro.loc[hv, split]:+.3f}  (v1 gap {macro.loc[xg, split] - macro.loc[hy, split]:+.3f})")
'''.split("\n")
c.source = "\n".join(lines[:at] + insert + lines[at:])
nbformat.write(nb, p); print("patched")
