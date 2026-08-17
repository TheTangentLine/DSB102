# DSB102 — Lab Assessment Revision Guide
### Everything needed to solve `Practice_Version.ipynb` (and the real A1)

---

## The one-paragraph version

The assessment is **90 minutes, 100 marks, closed-AI, open-slides**, and it is not testing whether you can invent anything. **Part A (40)** and **Part B (40)** are almost entirely *fill in the blank* — you need the sklearn method names in muscle memory, nothing more. Only **Task 1 of Part A is free-coding**, and it's four lines. The marks that actually separate people are the **three written answers (8 + 8 + 20 = 36 marks, over a third of the paper)**, and those are graded on whether you **cite your own printed numbers**. So the strategy is: get all the code running in the first 35 minutes, print more than you're asked to, then spend real time writing.

> **Master metaphor — a driving test, not a road trip.** Nobody is checking whether you can find a clever route. They're checking that you indicate, check the mirror, and don't stall. `.fit()`, `.predict()`, `.predict_proba(X)[:, 1]`, `r2_score`, `confusion_matrix`, `roc_auc_score` — these are the mirror checks. Drill them until they're automatic, because the marks you can actually *win* are in the written answers, and you only get to those if the code took you 35 minutes instead of 70.

---

# PART 1 — The blanks, drilled

Every `______` in the practice paper, and every one likely in the real one. Cover the right column.

## The universal five

| Blank | Answer |
|---|---|
| Create a model | `LinearRegression()`, `LogisticRegression(max_iter=5000)` |
| Train it | `.fit(X_train, y_train)` |
| Hard predictions | `.predict(X_test)` |
| **Probabilities** | `.predict_proba(X_test)[:, 1]` |
| Fitted coefficients | `.coef_` (note the **trailing underscore**) |

> ⚠️ **The trailing underscore is not decoration.** sklearn's convention: anything learned *from the data* ends in `_` — `coef_`, `intercept_`, `feature_importances_`, `classes_`, `oob_score_`. Anything you *set* has no underscore (`max_iter`, `n_estimators`). Writing `model.coef` returns an AttributeError, and in a 90-minute paper that's three minutes gone.

## Metrics

| Blank | Answer | Takes |
|---|---|---|
| R² | `r2_score(y_true, y_pred)` | predictions |
| MSE | `mean_squared_error(y_true, y_pred)` | predictions |
| RMSE | `np.sqrt(mean_squared_error(y_true, y_pred))` | predictions |
| Accuracy | `accuracy_score(y_true, y_pred)` | **labels** |
| Confusion matrix | `confusion_matrix(y_true, y_pred)` | **labels** |
| All metrics | `classification_report(y_true, y_pred)` | **labels** |
| ROC curve | `roc_curve(y_true, y_prob)` → `fpr, tpr, thresholds` | **probabilities** |
| AUC | `roc_auc_score(y_true, y_prob)` | **probabilities** |

> ⚠️ **The single most common blank-filling error: feeding labels to `roc_curve` / `roc_auc_score`.** They need `predict_proba(...)[:, 1]`, not `predict(...)`. It won't crash — it returns a degenerate three-point curve and a meaningless AUC, so you lose the marks silently. **Rule: anything with "ROC" or "AUC" in the name eats probabilities. Everything else eats labels.**
>
> ⚠️ **Argument order is always `(y_true, y_pred)`.** Backwards is silently wrong for the confusion matrix (it transposes it) and silently right for accuracy, which is worse, because you'll never learn the habit.

## pandas blanks

| Blank | Answer |
|---|---|
| Coefficients as a labelled Series | `pd.Series(model.coef_, index=X_train.columns)` |
| Rank by absolute value | `.abs()` |
| Top 3 | `.nlargest(3)` |
| Bottom 3 | `.nsmallest(3)` |
| Sort | `.sort_values(ascending=False)` |

---

# PART 2 — Part A solved (Linear Regression, 40 marks)

## Task 1 — free coding (16 marks)

This is the only cell with no scaffolding. **Learn these six lines verbatim.**

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

model = LinearRegression()                 # the variable MUST be called `model`
model.fit(X_train, yr_train)
y_pred = model.predict(X_test)

test_r2 = r2_score(yr_test, y_pred)
print(f"Test R² = {test_r2:.3f}")
```

> ⚠️ **The instructions name the variable `model` because later cells import it.** Call it `lin` or `lm` and Task 2 throws a `NameError`. Read variable-name instructions literally — they're there because the paper is chained.
>
> ⚠️ **`yr_` is regression, `yc_` is classification.** The setup cell creates both. Fitting `LinearRegression` on `yc_train` runs perfectly happily and gives you nonsense.

**Result:** `Test R² = 0.747`

## Task 2 — top 3 by |coefficient| (16 marks)

```python
coefs = pd.Series(model.coef_, index=X_train.columns)   # (1) model.coef_
top3  = coefs.abs().nlargest(3)                         # (2) .abs()

print("Top 3 features by |coefficient|:")
print(coefs[top3.index].round(4))
```

**Result — the full coefficient table (print all five; it costs nothing and arms your written answers):**

| Feature | Coefficient | Rank by \|coef\| |
|---|---|---|
| **weather_index** | **+2.1793** | **1** |
| **connections** | **+1.5094** | **2** |
| **ontime_history** | **−0.1280** | **3** |
| dep_hour | −0.0448 | 4 |
| distance_km | +0.0100 | 5 |

Intercept: 5.8136. Test RMSE: 7.44 minutes.

> 🔍 **The trap hiding in this task, and the thing worth an extra sentence anywhere you can fit it.**
>
> `distance_km` has the **smallest** coefficient of all five (0.0100) and ranks dead last. But distance runs from 150 to 4000 km, with a standard deviation of **1098**, while weather runs 0–10 with an sd of 2.84. Standardise the features first and the ranking inverts completely:
>
> | Feature | Raw coef | **Coef per 1 sd** |
> |---|---|---|
> | distance_km | 0.0100 | **+11.00** 🏆 |
> | weather_index | 2.1793 | +6.25 |
> | ontime_history | −0.1280 | −1.75 |
> | connections | 1.5094 | +1.24 |
> | dep_hour | −0.0448 | −0.23 |
>
> **`distance_km` is by far the biggest real driver of delay and the question's own ranking method puts it last.** The reason is that a coefficient is "effect per **one unit**", and one kilometre is a trivially small amount of distance while one point of weather index is a tenth of the whole scale.
>
> **"Rank by |coefficient|" is only meaningful when features share a scale.** Say so in whichever written answer you can attach it to — it's exactly the point the marking scheme is fishing for, and almost nobody notices it.

## Interpretation (8 marks) — causation

**The question:** the weather index has a positive coefficient. Does severe weather *cause* delays, or could other factors explain the association? What would you check first?

> ⚠️ **The supplied `Practice_Version_Soln.ipynb` answers this with a paragraph about BMI and standardised coefficients.** That's a copy-paste error from a different dataset — it answers a *scale* question, not a *causation* question. Don't hand it back as-is. (Ironically its content is the `distance_km` point above, so it's useful in the wrong place.)

**Model answer:**

> No. The coefficient is an **association**, not a causal effect: it says that among flights in this dataset, a one-point higher weather index goes with about **2.18 minutes** more delay, holding the other four features fixed. Three things could produce that association without weather causing delay. **Confounding** — a variable driving both, such as season or airport, since winter routes may have both worse weather and more congestion. **Reverse causality** is implausible here but must be ruled out in general. And **omitted-variable bias** — the model only adjusts for the four other features it contains, so any cause of delay correlated with weather and absent from the model is loaded onto the weather coefficient.
>
> Before claiming causation I would check: whether plausible confounders (airport, season, time of year, aircraft type) are available and whether the coefficient survives their inclusion; whether the relationship is stable across subgroups; the size of the standard error and confidence interval, since a coefficient that isn't distinguishable from zero supports nothing; and above all whether the data came from an experiment or a natural experiment. **Observational data with five predictors cannot establish causation regardless of how significant the coefficient is** — "holding all else fixed" only holds the things you actually measured fixed.

**What earns the 8 marks:** the words *association not causation*, at least one named confounder specific to flights, the phrase *"holding the other predictors fixed"* correctly used, and a concrete check you would run. Citing the **2.18** figure is what turns a generic answer into a specific one.

---

# PART 3 — Part B solved (Classification, 40 marks)

## Task 1 — Logistic regression (16 marks)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

lr = LogisticRegression(max_iter=5000)     # (1)
lr.fit(X_train, yc_train)                  # (2)  note yc_, not yr_
y_pred_lr = lr.predict(X_test)             # (3)

print("Confusion Matrix:")
print(confusion_matrix(yc_test, y_pred_lr))                    # (4)
print(f"Accuracy:  {accuracy_score(yc_test, y_pred_lr):.3f}")  # (5)
```

**Result — accuracy 0.833**, confusion matrix:

```
[[47 11]
 [ 9 53]]
```

**Read it in sklearn's layout — actual on rows, predicted on columns, `[[TN, FP], [FN, TP]]`:**

| | Pred on-time (0) | Pred delayed (1) |
|---|---|---|
| **Actual on-time (0)** | TN = 47 | FP = 11 |
| **Actual delayed (1)** | **FN = 9** | TP = 53 |

- **Recall (of delayed flights caught):** 53 / 62 = **0.855**
- **Precision (of flags that were right):** 53 / 64 = **0.828**
- **9 delayed flights were missed. 11 on-time flights were flagged unnecessarily.**

> ⚠️ **`max_iter=5000` is in the template for a reason.** The features are unscaled and `distance_km` runs to 4000, so the default 100 iterations doesn't converge. If you ever hit a `ConvergenceWarning` in the real paper, raise `max_iter` — don't ignore it.

## Task 2 — ROC and AUC (16 marks)

```python
from sklearn.metrics import roc_curve, roc_auc_score

probs = lr.predict_proba(X_test)[:, 1]     # (1) PROBABILITIES, column 1
fpr, tpr, _ = roc_curve(yc_test, probs)    # (2)
auc = roc_auc_score(yc_test, probs)        # (3)
```

**Result: AUC = 0.930.**

**What to write if asked to interpret it:** pick one random delayed flight and one random on-time flight — there's a **93% chance the model gives the delayed one a higher score**. That's a threshold-free statement, which is why AUC is comparable across models. Accuracy 0.833 versus AUC 0.930 is the normal pattern: the ranking is better than the default 0.5 cut-off is exploiting.

> ⚠️ **`[:, 1]`, not `[:, 0]`.** Column 0 is P(on-time), column 1 is P(delayed). Take column 0 and your AUC comes out as **0.070** — if you see an AUC below 0.5, this is why, every time.

## Interpretation (8 marks) — which error is costlier

> ⚠️ **Again, the supplied solution answers this about diabetes screening and early intervention.** Copy-paste error. The structure of its argument is right; the domain is wrong.

**Model answer:**

> The **false negative** is more costly here. A missed delay means passengers are not warned: they arrive at the airport on time, miss connections, and the airline absorbs rebooking costs and complaints. A false positive means an on-time flight is flagged — a passenger checks their phone, perhaps leaves a little later, and is mildly inconvenienced. The two errors are **not symmetric**, so the 0.5 threshold, which implicitly assumes they are, is the wrong cut-off.
>
> I would therefore **lower the threshold** to flag more flights as at-risk, trading precision for recall. On my test set at 0.5 there were **9 false negatives and 11 false positives (accuracy 0.833, recall 0.855)**. Dropping the threshold to **0.3** takes false negatives to **5** and raises recall to **0.919**, at the cost of four extra false alarms (11 → 15) and no loss in overall accuracy (0.833). **Four extra warned-but-punctual passengers to correctly warn four more delayed ones is a good trade in this setting.**

**The numbers, if you want to show the sweep:**

| Threshold | TN | FP | **FN** | TP | Accuracy | Recall | Precision |
|---|---|---|---|---|---|---|---|
| 0.5 (default) | 47 | 11 | **9** | 53 | 0.833 | 0.855 | 0.828 |
| 0.4 | 46 | 12 | 7 | 55 | **0.842** | 0.887 | 0.821 |
| **0.3** | 43 | 15 | **5** | 57 | 0.833 | **0.919** | 0.792 |

```python
for t in [0.3, 0.4, 0.5]:
    p = (probs >= t).astype(int)
    print(t, confusion_matrix(yc_test, p).ravel(), accuracy_score(yc_test, p))
```

**What earns the 8 marks:** naming which error type is worse **and why in flight terms**, saying **lower** the threshold, and quoting **your own FN count before and after**. Two sentences of specifics beats a paragraph of theory.

> **Why "lower"?** The threshold is the bar a flight must clear to be called delayed. Lower the bar → more flights flagged → more true positives *and* more false positives → **recall up, precision down**. If you can't remember the direction, reason it out from the extreme: threshold 0 flags everything, catching every delay (recall = 1) and crying wolf constantly.

---

# PART 4 — Foundations (20 marks)

These are worth a fifth of the paper and take ten minutes. **Both explicitly demand your own numbers.**

## Question 1 — Bias–variance (10 marks)

*"Which of your two methods has higher variance, and why? Refer to the specific methods you used."*

**Model answer for this paper:**

> Both of my methods are **linear models with the same five predictors**, so neither is especially flexible — but of the two, the **logistic regression in Part B has slightly higher variance**. Linear regression is fitted in closed form by least squares and estimates 6 parameters; logistic regression estimates the same 6 by iterative maximum likelihood on a **binarised** response, which discards the magnitude of each delay and keeps only which side of the median it fell. Throwing away that information means each coefficient is pinned down by less signal, so the estimates would move more from one training sample to another.
>
> The more useful point is that **both are at the low-variance, higher-bias end of the spectrum**. With 480 training rows and 5 predictors, neither can chase noise: my test R² of **0.747** is essentially identical to my training R² of **0.741**, which is the signature of a model that is not overfitting. A more flexible method on this data — an unpruned decision tree, or KNN with K = 1 — would have far higher variance: it would fit the training set close to perfectly and generalise worse.

**What earns the marks:** the direction (**more flexible = higher variance, lower bias**), applied to **your two named methods**, not stated abstractly. If you have time, name a genuinely high-variance alternative for contrast — it proves you understand the axis rather than the sentence.

## Question 2 — Why a held-out test set (10 marks)

*"What would happen to your metrics if you'd evaluated on the training set instead?"*

> 🔍 **Read this before you answer, because the expected answer is not what happens here.**
>
> The template answer is "training metrics are optimistic, so they'd be higher." On this data they are **not**:
>
> | Metric | Train | Test |
> |---|---|---|
> | R² (Part A) | 0.741 | **0.747** |
> | Accuracy (Part B) | 0.854 | 0.833 |
>
> **The test R² is marginally higher than the training R².** Write "my training R² would be higher" without checking and you have contradicted your own output. Compute both — it's one extra line — and write the honest version, which is a better answer anyway.

**Model answer:**

> A metric computed on the training set is **optimistically biased**, because the model chose its parameters to fit exactly that data and will have fitted some of its noise along with the signal. The held-out test set played no part in fitting, so it estimates how the model performs on flights it has never seen — which is the only thing anyone cares about.
>
> In my case the gap is **negligible: training R² 0.741 versus test R² 0.747**, and training accuracy 0.854 versus test accuracy 0.833. The optimism is small here for a specific reason — an ordinary linear model with **5 predictors fitted on 480 observations** has very little capacity to overfit, and the data-generating relationship is close to linear, so there is almost nothing for it to memorise. The test R² sitting a hair *above* the training R² is not evidence of anything real; it is sampling noise on a 120-row test set.
>
> **The gap grows with model flexibility**, which is exactly why the held-out set is non-negotiable rather than optional. Had I fitted an unpruned decision tree, training R² would have been **1.000** — a perfect fit achieved by giving nearly every training flight its own leaf — while test performance collapsed. **A training metric cannot distinguish a model that has learned from a model that has memorised; only held-out data can.**

**What earns the marks:** the mechanism (fits noise as well as signal), the words **optimistic / overfitting / generalisation**, and **your actual numbers**. Noticing that the gap is tiny *and explaining why* is what pushes this from 7 to 10.

---

# PART 5 — The 90-minute plan

| Time | Do this |
|---|---|
| **0–5 min** | Run the setup cell. Read every task heading before writing anything. Note which variables the paper names (`model`, `lr`, `probs`). |
| **5–20** | Part A Tasks 1–2. Print the **full** coefficient table, plus **training** R² as well as test. |
| **20–35** | Part B Tasks 1–2. Print the confusion matrix, accuracy, AUC, and **training accuracy**. |
| **35–45** | Restart & Run All. Fix anything that breaks. Copy the printed numbers into a scratch cell. |
| **45–80** | Write the three answers, quoting numbers from the scratch cell. |
| **80–90** | Checklist. Restart & Run All one final time. |

> **Print more than you're asked for.** Training metrics, the whole coefficient table, precision and recall, a threshold sweep. It costs 60 seconds and it is the raw material for 36 marks of writing. A written answer with a number in it reads as evidence; the same answer without one reads as revision notes.

---

# PART 6 — Traps that cost marks

1. **Naming the variable something other than `model`** when the paper says `model`. Later cells break.
2. **`yr_` vs `yc_`.** Regression target vs classification target. Both exist; both run silently.
3. **`predict()` into `roc_curve`/`roc_auc_score`.** They need `predict_proba(X)[:, 1]`. No error, just a meaningless AUC.
4. **`[:, 0]` instead of `[:, 1]`.** AUC 0.070 instead of 0.930.
5. **Missing trailing underscore** on `coef_`, `intercept_`, `feature_importances_`.
6. **Reading the confusion matrix in the slides' orientation.** sklearn is actual × predicted: `[[TN, FP], [FN, TP]]`. Getting this backwards inverts your entire Part B interpretation.
7. **Ranking raw coefficients across differently-scaled features** without saying that's what you're doing. `distance_km` is the biggest driver here and ranks last.
8. **Generic written answers.** "Overfitting means the model fits training noise" with no numbers is a pass, not a distinction. Every rubric line here says *reference your own results*.
9. **Answering the causation question with a scale answer** (or vice versa) — read which one is actually being asked.
10. **Not running Restart & Run All at the end.** Out-of-order execution is the classic way to submit a notebook that worked on your screen and fails on the marker's.

---

# PART 7 — If the real paper uses different topics

The practice paper covers Weeks 2 and 4. The real one may draw on Weeks 3 or 5. The structure won't change — only the model constructor.

| Topic | Constructor | Distinctive blanks |
|---|---|---|
| **Ridge** (W3) | `Ridge(alpha=1.0)` | `RidgeCV(alphas=...)`, `.alpha_` for the chosen value |
| **Lasso** (W3) | `Lasso(alpha=0.1)` | `LassoCV(cv=5)`, `np.sum(model.coef_ != 0)` for features kept |
| **Scaling** (W3) | `StandardScaler()` | `.fit_transform(X_train)` then **`.transform(X_test)`** — never `fit` on test |
| **Cross-validation** (W3) | `cross_val_score(model, X, y, cv=5)` | `scoring='neg_mean_squared_error'` — sklearn **negates** MSE |
| **Naive Bayes / KNN** (W4) | `GaussianNB()`, `KNeighborsClassifier(n_neighbors=k)` | KNN **requires scaled features** |
| **Decision tree** (W5) | `DecisionTreeRegressor(max_depth=3)` | `.get_n_leaves()`, `ccp_alpha=`, `plot_tree(m, feature_names=X.columns)` |
| **Random forest** (W5) | `RandomForestClassifier(n_estimators=200)` | `max_features='sqrt'`, `.feature_importances_` |
| **Boosting** (W5) | `GradientBoostingRegressor(learning_rate=0.05)` | `.staged_predict(X_test)` |

**Which methods need scaling:** KNN and any regularised model (Ridge, Lasso, and sklearn's `LogisticRegression`, which is L2-penalised by default). **Trees, forests and boosting never need it.**

---

# Quick self-test

Cover the answers.

1. **What does `model.coef_` return, and what's the underscore for?** — *The fitted coefficients, one per feature, in column order. The trailing underscore is sklearn's marker for an attribute learned from data.*
2. **Which metrics take probabilities rather than labels?** — *`roc_curve` and `roc_auc_score`. Everything else — accuracy, confusion matrix, precision, recall — takes hard labels.*
3. **You get AUC = 0.07. What did you do?** — *Took `predict_proba(X)[:, 0]` instead of `[:, 1]`, so you scored the negative class.*
4. **In `[[47, 11], [9, 53]]`, how many delayed flights were missed?** — *Nine. sklearn is actual × predicted, so `[[TN, FP], [FN, TP]]` and FN = 9.*
5. **False negatives are worse. Raise or lower the threshold?** — *Lower. More flights get flagged, so recall rises and precision falls.*
6. **`weather_index` has a coefficient of 2.18 and `distance_km` has 0.01. Which matters more?** — *Can't tell from those numbers. Distance is measured in kilometres and spans 3850 of them; per standard deviation it contributes 11.0 minutes against weather's 6.3. Ranking raw coefficients is only valid on a common scale.*
7. **What would your training R² be if you fitted an unpruned decision tree?** — *1.000, or near it — one leaf per observation. Which is exactly why training metrics can't be used to compare models.*
8. **Why does the template say `max_iter=5000`?** — *The features are unscaled and `distance_km` reaches 4000, so the solver doesn't converge in the default 100 iterations.*
9. **Both your models are linear. How do you answer a bias–variance question?** — *Say so honestly, identify the marginally more flexible one and why, then contrast both against a genuinely high-variance method (unpruned tree, K=1 KNN) to show you understand the axis.*
10. **The question says "store it in a variable called `model`". Does that matter?** — *Yes. The next cell references `model.coef_` and will throw a `NameError` otherwise.*
