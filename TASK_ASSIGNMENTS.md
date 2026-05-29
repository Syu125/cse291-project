# Project Task Assignments & Coordination

## Person 1 — Feature Selection & Model Writeup
- [ ] Document final feature processing strategy (missing values, feature selection steps)
- [ ] Summarize logistic regression selection and final optimizations
- [ ] Report final evaluation scores
- [ ] Assist others as needed

## Person 1 — Feature Processing Strategy (Writeup)

**Final Feature Processing Steps:**

1. **Raw Data Loading**
   - Loaded peptide-variant intensity matrix from the MAESTRO TSV file.

2. **Feature Selection**
   - Selected only the `Peptide` column and columns ending with `_intensity_for_peptide_variant` as features.

3. **Metadata Parsing**
   - Extracted patient condition and sample ID from intensity column names.

4. **Control Removal**
   - Dropped columns for `Empty` and `Norm` controls (non-patient samples).

5. **Missing Value Handling**
   - Converted all zero intensities to `NaN` (missing), as zeros represent non-detection in MS data.
   - Removed peptides (features) missing in more than 50% of patient samples.
   - Remaining missing values were imputed with zero (non-detection).

6. **Transformation**
   - Applied `log1p` transformation to all intensities to reduce skew.

7. **Matrix Transposition**
   - Transposed the matrix so each row is a patient/sample and each column is a peptide feature.

8. **Task-Specific Splits**
   - Created three binary classification tasks (Healthy vs. Infected, Severe vs. Non-severe, Symptomatic non-COVID vs. COVID).
   - For each task: stratified train/test split (80/20), scaling (fit on train only), and saved processed data.

## Person 1 — Logistic Regression Model Selection & Evaluation

**Model Selection:**
- Compared L1, L2, and ElasticNet regularization for each task.
- Selected the best model by cross-validation balanced accuracy and model sparsity (nonzero coefficients).

**Best Models & Scores:**
```markdown
**Healthy vs. Infected**
- Best Model: L1 penalty, C=1.0
- Train/Test Split: 72/18
- Features: 35,882 (67 nonzero coefficients)
- Test Scores: Accuracy 1.00, Balanced Accuracy 1.00, Precision 1.00, Recall 1.00, F1 1.00, ROC-AUC 1.00

**Severe vs. Non-severe**
- Best Model: ElasticNet (C=0.1, l1_ratio=0.5)
- Train/Test Split: 34/9
- Features: 35,882 (32 nonzero coefficients)
- Test Scores: Accuracy 1.00, Balanced Accuracy 1.00, Precision 1.00, Recall 1.00, F1 1.00, ROC-AUC 1.00

**Symptomatic non-COVID vs. COVID**
- Best Model: L1 penalty, C=1.0
- Train/Test Split: 54/14
- Features: 35,882 (48 nonzero coefficients)
- Test Scores: Accuracy 1.00, Balanced Accuracy 1.00, Precision 1.00, Recall 1.00, F1 1.00, ROC-AUC 1.00
```

**Notes:**
- All tasks achieved perfect test set performance, likely due to strong signal or possible overfitting (small test sets).
- Feature selection via regularization resulted in sparse, interpretable models.

## Person 2 — Feature Importance
- [ ] Extract top features from logistic regression coefficients
- [ ] Share top feature list with Persons 3 & 4 (**must be done before they start**)
- [ ] Create heatmaps and histograms of feature intensities (Severe vs. Non-severe)

## Person 3 — Peptide/Spectrum Validation
- [ ] Gather PSM evidence for top 3 features using USI comparison tool
- [ ] Compare modified vs. unmodified spectra ("Shifted" cosine)
- [ ] Summarize protein identification (bipartite graph or table)

## Person 4 — Differential Abundance
- [ ] Analyze missing values for top peptides
- [ ] Perform t-test and Wilcoxon test for top peptides/variants
- [ ] Compare DA for modified vs. unmodified peptides
- [ ] Assess protein-level DA (majority rule)

---

### Key Coordination Point
- [ ] **Person 2 must identify and share the top features/peptides with Persons 3 & 4 before they begin their analyses.**
