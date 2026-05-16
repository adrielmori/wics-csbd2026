# Statistical Significance Summary

This file was generated automatically by `codes/4-estatistic-result.ipynb`.

## Test split

- Datasets available: 10
- Models available: 32
- Best mean macro-F1 model: `bertimbau_large + svc_rbf` (0.8243)
- Friedman status: ok
- Friedman p-value: 1.156e-19
- Complete datasets used in Friedman: 10
- Models significantly different from the best model after Holm-Bonferroni: none

### Paper-ready results sentence

On the test split, the Friedman test compared 32 models across 10 complete datasets using dataset-level macro-F1 values (chi-square = 161.9531, p = 1.156e-19). The model with the highest mean macro-F1 was `bertimbau_large + svc_rbf` (mean macro-F1 = 0.8243); none of the 31 post-hoc comparisons remained significant after Holm-Bonferroni correction at alpha = 0.05.

## Val split

- Datasets available: 10
- Models available: 32
- Best mean macro-F1 model: `bertimbau_large + svc_rbf` (0.8110)
- Friedman status: ok
- Friedman p-value: 4.776e-16
- Complete datasets used in Friedman: 10
- Models significantly different from the best model after Holm-Bonferroni: none

### Paper-ready results sentence

On the val split, the Friedman test compared 32 models across 10 complete datasets using dataset-level macro-F1 values (chi-square = 141.4335, p = 4.776e-16). The model with the highest mean macro-F1 was `bertimbau_large + svc_rbf` (mean macro-F1 = 0.8110); none of the 31 post-hoc comparisons remained significant after Holm-Bonferroni correction at alpha = 0.05.

## Paper-ready methodology text

To assess whether the observed differences among the classical models were statistically significant, we treated each dataset as the statistical unit and computed one macro-F1 score for every dataset-model pair. The main analysis was performed on the test split, while the validation split was analyzed only as a complementary check. We first applied the Friedman test to compare all models jointly across datasets. We then selected the model with the highest mean macro-F1 across datasets and compared it against each remaining model using two-sided Wilcoxon signed-rank tests. The resulting p-values were adjusted with the Holm-Bonferroni correction to control the family-wise error rate. All statistical decisions used a significance level of alpha = 0.05. This design avoids treating individual predictions as independent observations and instead bases the inference on dataset-level performance.
