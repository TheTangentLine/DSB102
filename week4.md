# DSB102 — Week 4: Classification
### Complete study guide (lecture slides + solved lab)

---

## The one-paragraph version

Weeks 2–3 predicted a **number**. Week 4 predicts a **category** — and the honest version of that task is not "which class?" but "**what's the probability of each class?**". Once you have a probability you can choose a threshold, and that choice belongs to the problem, not the algorithm: missing a cancer is not the same kind of mistake as an unnecessary follow-up scan. So this week has three parts. **Three classifiers** — logistic regression (models the log-odds as linear), naive Bayes (assumes features are independent given the class), and KNN (asks the nearest neighbours to vote). **A vocabulary of errors** — the confusion matrix, precision, recall, specificity, F1. And **threshold-free evaluation** — ROC curves and AUC, which judge a model across *every* threshold at once.

> **Master metaphor — a medical test.** The classifier is the assay; it returns a number. The **threshold** is where the clinic decides to call it positive, and that's a policy decision reflecting the cost of each mistake. The **confusion matrix** is the audit of what the clinic actually got right and wrong. The **ROC curve** is the assay's quality report across every possible cut-off, so you can compare two assays before committing to one. Confusing the assay with the policy is the single biggest conceptual error in this topic.

**The thread from Week 2:** logistic regression is still $\beta_0 + \beta_1X_1 + \cdots$. All that changed is what gets wrapped around it and how it's fitted.

---

## Setup — imports used throughout

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, precision_score, recall_score,
                             roc_curve, roc_auc_score)
```

---

# PART 1 — What classification is

Given a feature vector $X$ and a **qualitative** response $Y$ taking values in a set $\mathcal{C}$, build a function $C(X)$ that predicts $Y$, with $C(X) \in \mathcal{C}$.

Qualitative variables take values in an **unordered** set — that word matters. There's no sense in which "malignant" is greater than "benign", so you can't just regress on the labels.

**The slides make a point you should internalise:** we're often more interested in the **probability** that $X$ belongs to each category than in the hard label. *"It is more valuable to have an estimate of the probability that an insurance claim is fraudulent than a classification fraudulent or not."*

> **Why?** A hard label throws away everything you need to make a decision. "Fraudulent" tells an investigator nothing about whether to open a case; "87% likely fraudulent" tells them to open it today, and "51%" tells them to gather more evidence. **Probabilities carry the confidence; labels discard it.** Every classifier in this course has a `.predict_proba()` — use it.

## 1.1 Why not just use linear regression?

Code $Y = 1$ for Yes, $0$ for No, run OLS, and classify as Yes if $\hat{Y} > 0.5$. Does it work?

**Partly.** For a binary outcome, linear regression is a *reasonable* classifier — and there's a real reason: in the population, $E(Y|X=x) = \Pr(Y=1|X=x)$. The conditional mean of a 0/1 variable **is** a probability. So regression looks perfect for the job.

**The problem:** linear regression will happily produce fitted values **below 0 or above 1**, which cannot be probabilities.

> **Metaphor:** a straight line has no idea it's supposed to be a probability. Extend it far enough and it will confidently forecast a **−12% chance of default**. Logistic regression takes the same linear score and bends it so it can never leave $[0,1]$ — the sigmoid is a straight line squeezed through a funnel that has walls at 0 and 1.

For **more than two classes**, ordinary linear regression breaks down completely: coding three categories as 1, 2, 3 imposes an ordering and equal spacing that don't exist.

---

# PART 2 — Logistic regression

## 2.1 The three equivalent forms

Write $p(X) = \Pr(Y=1|X)$. The **logistic function**:

$$p(X) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}}$$

No matter what $\beta_0$, $\beta_1$ or $X$ are, $p(X)$ lands strictly between 0 and 1.

**Odds:**
$$\frac{p(X)}{1 - p(X)} = e^{\beta_0 + \beta_1 X}$$

**Log-odds (logit)** — take logs and rearrange:
$$\log\left(\frac{p(X)}{1-p(X)}\right) = \beta_0 + \beta_1 X$$

**This last line is the punchline of the whole topic.** The right-hand side is *exactly* the Week 2 linear model. Logistic regression is linear regression applied to the **log-odds** rather than to $Y$ directly. This monotone transformation is called the **logit**.

| Scale | Range | Interpretation of $\beta_1$ |
|---|---|---|
| Log-odds | $(-\infty, \infty)$ | **Additive**: +$\beta_1$ per unit of X. Linear here. |
| Odds | $(0, \infty)$ | **Multiplicative**: odds × $e^{\beta_1}$ per unit of X |
| Probability | $(0, 1)$ | **Non-linear**: the effect depends on where you start |

> **Metaphor — three currencies for one amount.** Log-odds is the currency where the model is a straight line, so it's where the maths is easy. Probability is the currency humans understand. Odds sits between them and is how gamblers and epidemiologists talk. Same quantity, three exchange rates — and **the coefficient only means "a constant effect" in the log-odds currency.**

> **The consequence people miss:** in probability terms the *same* coefficient does very different things depending on where you are. Going from a balance of \$1000 to \$2000 moves the default probability from 0.006 to 0.586 — a huge jump. Going from \$3000 to \$4000 barely moves it, because it's already pinned near 1. **A logistic coefficient is not "the change in probability".** The sigmoid is steepest at $p = 0.5$ and flattens at both ends.

## 2.2 Maximum likelihood

Least squares doesn't apply here. Instead we maximise the **likelihood**:

$$\ell(\beta_0, \beta_1) = \prod_{i:\,y_i=1} p(x_i) \times \prod_{i:\,y_i=0} (1 - p(x_i))$$

This is the probability of observing exactly the 0s and 1s you did observe, given $\beta_0, \beta_1$. Choose the parameters that make your actual data as unsurprising as possible.

**The slides' worked example** — three customers:

| Customer | $y_i$ | $p(x_i)$ | Contribution |
|---|---|---|---|
| 1 | 1 | 0.8 | $p(x_1) = 0.8$ |
| 2 | 0 | 0.1 | $1 - p(x_2) = 0.9$ |
| 3 | 0 | 0.3 | $1 - p(x_3) = 0.7$ |

$$\ell = 0.8 \times 0.9 \times 0.7 = 0.504$$

**Read the pattern:** for a customer who *did* default you take $p$; for one who *didn't* you take $1-p$. Every term is "the probability the model assigned to what actually happened". Multiply them, then maximise.

> **Metaphor:** you're grading a forecaster. Every day they gave a probability of rain; you look up what actually happened and collect the probability they assigned to the real outcome. Multiply all those together. A forecaster who said 90% on rainy days and 5% on dry days scores enormously higher than one who said 50% every day. Maximum likelihood tunes the forecaster to maximise that score.

**Why multiply?** Because the observations are assumed independent, and independent probabilities multiply. In practice software maximises the **log**-likelihood — sums are easier than products, and multiplying hundreds of numbers below 1 underflows to zero. Maximising the log of a thing maximises the thing, since log is monotone.

> **Connection worth knowing:** the negative log-likelihood *is* **cross-entropy loss** — the standard loss function for classification in neural networks. You are learning the same objective deep learning uses, in its simplest setting.

## 2.3 The credit card example

| Coefficient | Estimate | Std. Error | Z-statistic | P-value |
|---|---|---|---|---|
| Intercept | −10.6513 | 0.3612 | −29.5 | < 0.0001 |
| balance | 0.0055 | 0.0002 | 24.9 | < 0.0001 |

**Predicting at balance = \$1000:**
$$\hat{p} = \frac{e^{-10.6513 + 0.0055 \times 1000}}{1 + e^{-10.6513 + 0.0055 \times 1000}} = \frac{e^{-5.15}}{1+e^{-5.15}} = 0.006$$

**At \$2000:**
$$\hat{p} = \frac{e^{-10.6513 + 0.0055 \times 2000}}{1 + e^{-10.6513+0.0055 \times 2000}} = \frac{e^{0.35}}{1+e^{0.35}} = 0.586$$

Double the balance, and the probability goes from 0.6% to 58.6%. **That's the non-linearity in action** — the same $1000 increase did almost nothing in one region and everything in another.

> ⚠️ **Z-statistic, not t-statistic.** Logistic regression reports **z** because maximum likelihood estimates are asymptotically normal, whereas OLS with normal errors gives an exact t-distribution. Interpretation is identical: estimate ÷ standard error, and |z| > 2 is roughly significant. Just don't write "t" in an exam answer about logistic regression.

## 2.4 Multiple logistic regression

$$\log\left(\frac{p(X)}{1-p(X)}\right) = \beta_0 + \beta_1X_1 + \cdots + \beta_pX_p$$

| Coefficient | Estimate | Std. Error | Z | P-value |
|---|---|---|---|---|
| Intercept | −10.8690 | 0.4923 | −22.08 | < 0.0001 |
| balance | 0.0057 | 0.0002 | 24.74 | < 0.0001 |
| income | 0.0030 | 0.0082 | 0.37 | **0.7115** |
| student[Yes] | **−0.6468** | 0.2362 | −2.74 | 0.0062 |

**Two things to read here:**

1. **`income` is not significant** (p = 0.71). Once balance is known, income adds nothing — its coefficient is smaller than its own standard error.
2. **`student[Yes]` is negative**, so students are *less* likely to default **holding balance fixed**. Yet in the raw data students default *more* often. Both are true: students carry higher balances, and the balance drives the risk. Compare two people with the *same* balance and the student is safer.

> This is **confounding** — the Week 2 "holding all else fixed" clause, biting hard. It's also the classic exam question on this slide: *"students default more, but the coefficient is negative — explain."* The answer is that the marginal and conditional relationships genuinely have opposite signs, and only the conditional one is what the coefficient reports.

```python
lr = LogisticRegression(max_iter=1000).fit(X_tr_sc, y_tr)

y_pred = lr.predict(X_te_sc)                  # hard 0/1 labels at threshold 0.5
y_prob = lr.predict_proba(X_te_sc)[:, 1]      # P(class 1) — column 1 is the positive class

# Coefficients live on the LOG-ODDS scale; exponentiate for odds ratios
odds_ratios = np.exp(lr.coef_[0])

# For the slides' coefficient tables with z-stats and p-values, use statsmodels:
# import statsmodels.api as sm
# sm.Logit(y, sm.add_constant(X)).fit().summary()
```

> ⚠️ **`predict_proba` returns two columns.** Column 0 is P(class 0), column 1 is P(class 1). They sum to 1. **You almost always want `[:, 1]`** — getting this wrong silently inverts your ROC curve.
>
> ⚠️ **sklearn's `LogisticRegression` applies L2 regularisation by default** (`C=1.0`, where smaller C means *more* penalty). This is not textbook logistic regression. It's usually helpful — it stops coefficients exploding when classes are nearly separable — but it means **you must scale your features first**, exactly as in Week 3. For unpenalised textbook estimates use `penalty=None`, or use statsmodels.

## 2.5 More than two classes *(flagged [Advanced])*

For $K > 2$ classes, logistic regression generalises to **multinomial (softmax) regression**. Each class gets its own linear score $z_k = \beta_{k0} + \beta_{k1}x_1 + \cdots + \beta_{kp}x_p$, and softmax converts the $K$ scores into probabilities summing to 1:

$$\Pr(Y = k \mid X = x) = \frac{e^{z_k}}{\sum_{l=1}^{K} e^{z_l}}$$

Classify to the class with the highest probability. Fitted by maximum likelihood, exactly as before — **this is the cross-entropy loss, and softmax is the standard output layer of multi-class neural networks.**

> **Metaphor:** each class writes down an enthusiasm score. Softmax exponentiates (making everything positive and amplifying the leader) and then divides by the total so the scores become shares of a pie. The exponential is what makes it *soft*-max rather than hard max: a runner-up with a close score still keeps meaningful probability.

---

# PART 3 — Measuring classification performance

## 3.1 The confusion matrix

| | Actual Positive | Actual Negative |
|---|---|---|
| **Predicted Positive** | True Positive (TP) | False Positive (FP) |
| **Predicted Negative** | False Negative (FN) | True Negative (TN) |

| Metric | Formula | Question it answers |
|---|---|---|
| **Recall** (Sensitivity, TPR) | $\dfrac{TP}{TP+FN}$ | Of all the real positives, how many did I catch? |
| **Specificity** (TNR) | $\dfrac{TN}{TN+FP}$ | Of all the real negatives, how many did I correctly clear? |
| **Precision** | $\dfrac{TP}{TP+FP}$ | Of everything I flagged, how much was real? |
| **Type I error** (FPR) | $\dfrac{FP}{FP+TN}$ | False alarm rate $= 1 -$ specificity |
| **Type II error** (FNR) | $\dfrac{FN}{FN+TP}$ | Miss rate $= 1 -$ recall |
| **F1 score** | $\dfrac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of the two |

**How to keep precision and recall straight — look at the denominator:**
- **Recall** divides by the **actual** positives (the column). *Did I find them all?*
- **Precision** divides by the **predicted** positives (the row). *Was I right when I shouted?*

> **Metaphor — fishing with a net.** **Recall** is what fraction of the fish in the lake you caught. **Precision** is what fraction of your haul is actually fish rather than boots and weed. Use a bigger net and recall rises while precision falls. **You can trivially max out either one alone** — catch everything (recall = 1, precision terrible) or catch one guaranteed fish (precision = 1, recall terrible). That's why F1 exists: it's the harmonic mean, which stays low unless *both* are decent. A model with precision 1.0 and recall 0.01 has F1 ≈ 0.02, not 0.5.

⚠️ **Accuracy is dangerous with imbalanced classes.** If 99% of transactions are legitimate, predicting "legitimate" every time scores **99% accuracy** and catches zero fraud. Always look at the confusion matrix, never accuracy alone.

```python
print(confusion_matrix(y_te, y_pred))
print(classification_report(y_te, y_pred, target_names=["Benign", "Malignant"]))

ConfusionMatrixDisplay.from_predictions(
    y_te, y_pred, display_labels=["Benign", "Malignant"], colorbar=False)
plt.show()

# Individual metrics
print(precision_score(y_te, y_pred), recall_score(y_te, y_pred))
```

> ⚠️ **sklearn's confusion matrix layout is the transpose of the slides'.** sklearn puts **actual on the rows, predicted on the columns**, ordered `[0, 1]`, so it reads:
>
> `[[TN, FP], [FN, TP]]`
>
> The slides put predicted on the rows. Same information, flipped. Read the axis labels before you interpret any cell — mixing this up turns your FP into your FN.

## 3.2 Varying the threshold

The 0.5 cut-off is a **default, not a law**. You can move it anywhere in $[0,1]$ to trade the two error types against each other.

**From the slides' figure:** as the threshold rises from 0 to 0.5, the false positive rate falls and the false negative rate climbs. **A threshold of 0.5 minimises the overall error rate** — expected, since the Bayes classifier uses 0.5 and has the lowest possible overall error. **To reduce false negatives you may want a threshold of 0.1 or less.**

> **The crucial idea:** 0.5 is optimal *only if* a false positive and a false negative cost the same. In cancer screening they don't — a missed tumour costs a life, a false alarm costs a follow-up scan. **The threshold is where your loss function enters the model, and it's the one part of the pipeline the algorithm cannot choose for you.**

```python
y_prob = lr.predict_proba(X_te_sc)[:, 1]

y_pred_low = (y_prob >= 0.3).astype(int)     # more willing to call it positive
print(confusion_matrix(y_te, y_pred_low))

# Sweep the whole range to see the trade-off
for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
    p = (y_prob >= t).astype(int)
    print(f"t={t}: precision {precision_score(y_te, p):.3f}  "
          f"recall {recall_score(y_te, p):.3f}")
```

> **Note what `.predict()` really is.** It's just `predict_proba()[:, 1] >= 0.5`. There is no separate model for hard labels. Thresholding manually isn't a hack — it's taking back a decision sklearn made on your behalf.

## 3.3 ROC and AUC

The **ROC curve** plots **true positive rate against false positive rate** as the threshold sweeps across every possible value. **AUC** is the area under it, summarising performance across all thresholds at once. **Higher AUC is better.**

| AUC | Meaning |
|---|---|
| 1.0 | Perfect separation |
| 0.9+ | Excellent |
| 0.5 | Useless — the diagonal, equivalent to coin-flipping |
| < 0.5 | Worse than chance (usually means your labels or probability column are flipped) |

> **What AUC actually means, concretely:** pick one random positive case and one random negative case. **AUC is the probability the model gives the positive one a higher score.** That's a genuinely threshold-free statement, which is why AUC is comparable across models with different calibration.

> **Metaphor:** the confusion matrix is a photograph of the model at one setting. The ROC curve is the **full spec sheet** across every setting. A camera that takes one great photo at one shutter speed and unusable ones everywhere else is a worse camera than one that's good throughout — even though a single photo might not show it.

```python
fpr, tpr, thresholds = roc_curve(y_te, y_prob)      # note: PROBABILITIES, not labels
auc = roc_auc_score(y_te, y_prob)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, lw=2, label=f"Logistic  AUC={auc:.3f}")
ax.plot([0, 1], [0, 1], "k--", label="Random guess")
ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate (Recall)")
ax.legend()
plt.show()
```

> ⚠️ **Feed `roc_curve` probabilities, not predicted labels.** Pass hard 0/1 labels and you get a degenerate three-point curve and a meaningless AUC — a very common and easy-to-miss bug.
>
> **When AUC misleads:** with severe class imbalance, AUC stays flatteringly high because the huge TN count keeps FPR low. For rare-positive problems (fraud, disease screening) the **precision-recall curve** is more informative.

---

# PART 4 — Naive Bayes

**The assumption:** features are **independent given the class**. Instead of modelling one complicated joint distribution, estimate each feature's distribution separately and multiply:

$$f_k(x) = f_{k1}(x_1) \times f_{k2}(x_2) \times \cdots \times f_{kp}(x_p)$$

Combined with Bayes' theorem:

$$\Pr(Y = k \mid X = x) = \frac{\pi_k \times f_{k1}(x_1) \times \cdots \times f_{kp}(x_p)}{\sum_{l=1}^{K} \pi_l \times f_{l1}(x_1) \times \cdots \times f_{lp}(x_p)}$$

where $\pi_k$ is the **prior** — how common class $k$ is before you look at any features.

**In plain English (the slides):** for each class, multiply how "typical" the observation's value is for every feature under that class, then pick the class with the highest score.

> **Metaphor — a panel of narrow specialists.** One expert only looks at tumour radius, another only at texture, another only at smoothness. None of them talks to the others. Each says "under the malignant hypothesis, this value is fairly typical / very unusual". You multiply their verdicts and go with the winning class. It's **naive** because the experts ignore each other — in reality radius and perimeter are practically the same measurement, so their evidence gets **double-counted**.

**Why it works despite being wrong:** double-counting distorts the *probabilities* (they come out overconfident, pushed toward 0 and 1), but the **ranking** of classes often survives. And classification only needs the argmax to be right. So naive Bayes is frequently a good *classifier* and a poor *probability estimator*.

**When to use it:** the slides say when **p is very large**. It needs only $O(p)$ parameters instead of $O(p^2)$ for a full covariance matrix, so it's fast, works with tiny training sets, and is a classic baseline for text classification (thousands of word features).

```python
nb = GaussianNB().fit(X_tr, y_tr)         # Gaussian: assumes each feature is normal within each class
y_prob_nb = nb.predict_proba(X_te)[:, 1]

print(nb.class_prior_)     # the priors, estimated from class frequencies
print(nb.theta_)           # per-class, per-feature means
print(nb.var_)             # per-class, per-feature variances

# Variants for other data types:
#   MultinomialNB — counts (word frequencies)
#   BernoulliNB   — binary features
```

> **Does GaussianNB need scaled features?** **No — and this is a genuine property, not a rule of thumb.** It fits a separate mean and variance per feature per class, so standardising each feature is an affine change that cancels out of the calculation. I verified this: with `var_smoothing` set very small, raw and scaled data give *identical* accuracy (0.9474 both). The small difference you'd otherwise see comes only from `var_smoothing`, a numerical-stability term scaled to the largest feature variance. Contrast with **KNN and regularised logistic regression, which absolutely do need scaling.**

---

# PART 5 — From the Bayes classifier to KNN

## 5.1 The Bayes classifier — the unattainable gold standard

The **Bayes classifier** assigns each observation to the class with the highest true conditional probability $\Pr(Y=j \mid X=x_0)$. It achieves the lowest possible test error — the **Bayes error rate**.

**The catch: for real data we never know the true conditional distribution**, so the Bayes classifier cannot be computed. It's a benchmark, not a method.

> **Metaphor:** the Bayes error rate is the *speed of light* of your problem. It's the irreducible floor set by genuine overlap between the classes — two patients with identical measurements and different outcomes. No algorithm can go below it. It's the classification twin of the noise floor you hit in Week 2, when test RMSE ≈ 1.53 against a true σ of 1.5.

**KNN's idea:** if we can't know those conditional probabilities, **estimate them from nearby training data.**

## 5.2 The KNN algorithm

1. Choose $K \in \mathbb{Z}^+$ and a test point $x_0$.
2. Find the $K$ nearest neighbours $\mathcal{N}_0$ of $x_0$ in the training set, by distance.
3. Estimate the probability as the fraction of those neighbours in each class:
$$\Pr(Y = j \mid X = x_0) = \frac{1}{K}\sum_{i \in \mathcal{N}_0} I(y_i = j)$$
4. Classify to the class with the highest estimate — a **majority vote**.

> **Metaphor:** you move to a new street and want to guess which way the neighbourhood votes. Ask the three nearest houses. Ask thirty and you're really surveying the whole suburb, losing anything distinctive about your street. Ask one and you learn only about the eccentric next door. **K controls how far "local" reaches.**

**KNN is non-parametric** — it fits no equation and estimates no coefficients. "Training" is just memorising the data; all the work happens at prediction time.

## 5.3 Choosing K — the bias-variance trade-off, again

| K | Boundary | Bias / Variance | Risk |
|---|---|---|---|
| **K = 1** | Wiggly, very flexible | Low bias, **high variance** | **Overfitting** |
| **K = 7** | Smooth, balanced | Balanced | — found by cross-validation |
| **K = n** | Straight/flat, very rigid | **High bias**, low variance | **Underfitting** |

**As K increases, bias ↑ and variance ↓.** Choose K by cross-validation.

> **Note the direction, because it's the reverse of what people expect.** Large K = *less* flexible. It's the opposite of λ in Week 3 only in labelling, not in substance: both are dials along the same U-shaped curve from slide 23 of Week 3. K = n predicts the majority class everywhere — the classification analogue of the null model.

```python
knn = KNeighborsClassifier(n_neighbors=5).fit(X_tr_sc, y_tr)   # SCALED data
y_prob_knn = knn.predict_proba(X_te_sc)[:, 1]

# Choose K properly — by cross-validation on the TRAINING set, never on the test set
from sklearn.pipeline import make_pipeline
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for k in [1, 3, 5, 10, 20, 50]:
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(k))
    print(k, round(cross_val_score(pipe, X_tr, y_tr, cv=cv).mean(), 4))
```

> ⚠️ **KNN is distance-based, so scaling is mandatory.** Without it, a feature measured in the hundreds (mean area ≈ 650) completely swamps one measured in fractions (mean smoothness ≈ 0.1) — the "distance" becomes almost entirely area. On the lab data this costs real accuracy: **0.974 scaled → 0.930 unscaled.**
>
> **Use `StratifiedKFold` for classification**, so each fold keeps the class proportions. Plain `KFold` can hand you a fold with almost no positives.

---

# PART 6 — The slides' worked example

Predicting default from balance: $\hat{\beta}_0 = -10.65$, $\hat{\beta}_1 = 0.0055$. Predict for **balance = \$1,500**.

**Step 1 — the linear score (log-odds):**
$$\log\left(\frac{p}{1-p}\right) = -10.65 + 0.0055 \times 1500 = -2.4$$

**Step 2 — convert to probability:**
$$\hat{p} = \frac{e^{-2.4}}{1 + e^{-2.4}} \approx \frac{0.091}{1.091} \approx 0.083$$

**An 8.3% probability of default.**

**Step 3 — decision rule at threshold 0.5:** since $0.083 < 0.5$, classify as **No default**.

**Interpretation:**
- Each extra \$1 of balance raises the **log-odds** by 0.0055.
- **Odds ratio:** $e^{0.0055} \approx 1.0055$ — each extra \$1 multiplies the **odds** of default by about 1.0055.

> **Scale that up, because per-dollar it looks negligible.** Over \$100 the odds multiply by $e^{0.55} \approx 1.73$; over \$500, by $e^{2.75} \approx 15.6$. **Odds ratios compound multiplicatively**, which is why a coefficient that looks like a rounding error produces the dramatic S-curve on slide 5.

```python
def logistic_predict(b0, b1, x):
    log_odds = b0 + b1 * x
    return np.exp(log_odds) / (1 + np.exp(log_odds))    # = 1 / (1 + exp(-log_odds))

print(logistic_predict(-10.65, 0.0055, 1500))    # 0.0832
print(np.exp(0.0055))                            # odds ratio per $1: 1.0055
print(np.exp(0.0055 * 500))                      # per $500: 15.6
```

---

# PART 7 — The lab, worked through

**Dataset:** sklearn's breast cancer data — **569 samples, 30 features**, split 80/20 stratified with `random_state=17`.

```python
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)

# sklearn ships 0=malignant, 1=benign. The lab FLIPS this so Malignant=1.
y = pd.Series(1 - data.target, name="Malignant")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=17, stratify=y)   # stratify keeps class balance

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr)     # fit on TRAIN
X_te_sc = sc.transform(X_te)         # transform only — never fit on test
```

**Class balance:** Benign 357, Malignant 212.

> **Why the flip matters.** With Malignant = 1, "false negative" means *a missed cancer* — the error the exercises care about. Leave sklearn's default coding and every FN/FP statement in the lab inverts. **Always check which class is the positive one before interpreting a confusion matrix.**
>
> **Note the scaler discipline** — `fit_transform` on train, `transform` on test. This is the leak-free version of the mistake flagged in Week 3.

## Part 1 — Logistic regression

**Accuracy 0.965**, AUC **0.9974**. Confusion matrix (sklearn layout, actual × predicted):

| | Pred Benign | Pred Malignant |
|---|---|---|
| **Actual Benign** | 71 | 1 |
| **Actual Malignant** | **3** | 39 |

**Three malignant cases were missed.**

### Exercise 1 — lowering the threshold to 0.3

| | Pred Benign | Pred Malignant |
|---|---|---|
| **Actual Benign** | 70 | 2 |
| **Actual Malignant** | **1** | 41 |

| Threshold | Precision (Malignant) | Recall (Malignant) | Accuracy |
|---|---|---|---|
| 0.5 | 0.975 | 0.929 | 0.965 |
| **0.3** | 0.953 | **0.976** | 0.974 |

**Task 3 answer:** two extra malignant cases caught (3 misses → 1). The cost is one extra false alarm (1 → 2 benign flagged as malignant). **In screening that's an outstanding trade** — a follow-up biopsy versus a missed cancer.

> **Notice something unusual:** accuracy *also* improved (0.965 → 0.974). That's not the norm — 0.5 minimises overall error in general, and you'd normally expect to *sacrifice* accuracy for recall. Here the class imbalance and this particular split make 0.3 better on both counts. **Don't generalise from it.** The reason to move the threshold is the asymmetric cost, and that argument holds even when accuracy drops.

## Part 2 — Naive Bayes

**Accuracy 0.939**, AUC **0.9897**.

**Exercise 2 — the correlation check.** Among the first six features:

| Pair | r |
|---|---|
| mean radius ↔ mean perimeter | **1.00** |
| mean radius ↔ mean area | **0.99** |
| mean perimeter ↔ mean area | **0.99** |
| mean smoothness ↔ mean compactness | 0.67 |

**Task 2:** the independence assumption is **badly violated**. Radius, perimeter and area are three ways of measuring the same circle — $\text{perimeter} = 2\pi r$ and $\text{area} = \pi r^2$ are *deterministic* functions of radius, which is why r = 1.00.

**Task 3 — why does it still work?** Naive Bayes counts that single piece of evidence three times, making it wildly overconfident. But overconfidence in the *same direction* doesn't change which class wins the argmax. The class-conditional means are far enough apart that the ranking survives. **Accuracy holds up; the probabilities are not to be trusted** — a distinction that matters if you ever want to threshold them.

## Part 3 — KNN

**K=5 test accuracy 0.974.**

**Exercise 3 — tuning K:**

| K | 1 | **3** | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| Test accuracy | 0.956 | **0.974** | 0.974 | 0.974 | 0.974 | 0.956 |

**Model comparison (Task 3):**

| Model | Test accuracy |
|---|---|
| KNN (K=3) | **0.974** |
| Logistic Regression | 0.965 |
| Naive Bayes | 0.939 |

### 🔍 Two problems with "best K = 3" worth knowing

**1. It's a four-way tie broken by luck.** K = 3, 5, 10 and 20 all score exactly 0.974. `np.argmax` returns the *first* maximum, so K=3 wins on array position, not merit.

**2. K was chosen on the test set** — the exact thing Week 3 warned against. Selecting a hyperparameter on your test data means that test score is no longer an unbiased estimate. Doing it properly, with 5-fold CV on the *training* set:

| K | 1 | 3 | **5** | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| CV accuracy | 0.9604 | 0.9670 | **0.9692** | 0.9604 | 0.9516 | 0.9473 |

**Cross-validation picks K = 5**, cleanly and without a tie. And there's the U-shape again — accuracy rises to K=5, then falls away as the model gets too rigid.

## Part 4 — ROC curves

**Exercise 4, task 2 — which classifier has the highest AUC?**

| Model | Test accuracy | **AUC** |
|---|---|---|
| **Logistic Regression** | 0.965 | **0.9974** 🏆 |
| Naive Bayes | 0.939 | 0.9897 |
| KNN (K=3) | **0.974** 🏆 | **0.9608** ⚠️ |

### 🔍 The most instructive result in the lab: the rankings disagree

**KNN has the best accuracy and the worst AUC.** Logistic regression has lower accuracy and a near-perfect AUC. This isn't a contradiction — it's the clearest possible demonstration of why the slides insist on evaluating with more than accuracy.

**Why KNN's AUC is poor:** with K=3, `predict_proba` can only return **four possible values** — 0, ⅓, ⅔, or 1. The ROC curve therefore has only a handful of points and is a coarse staircase rather than a smooth arc. KNN classifies well at the default threshold but has almost no ability to *rank* cases by confidence.

```python
for k in [3, 5, 20]:
    p = KNeighborsClassifier(k).fit(X_tr_sc, y_tr).predict_proba(X_te_sc)[:, 1]
    print(k, len(np.unique(p)), "distinct probabilities")
# 3 -> 4,  5 -> 5,  20 -> 16
```

**Which is why this matters clinically.** If you want to lower the threshold to catch more cancers — Exercise 1's whole point — KNN with K=3 gives you almost nothing to work with: the only thresholds that do anything are ⅓ and ⅔. Logistic regression gives a continuous, finely-graded probability you can cut anywhere.

**Task 3 — would you choose on AUC alone?** No. Also weigh: **recall at your operating threshold** (the number that actually reflects missed cancers); the **cost asymmetry** between error types; whether the probabilities are **usable for thresholding** (KNN's aren't); **interpretability** for clinical sign-off; and **calibration** — whether "0.8" really means 80%.

**Overall recommendation:** **logistic regression**. It gives up 0.9 percentage points of accuracy on this split — well within noise on 114 test samples — and in exchange offers near-perfect ranking, smooth thresholdable probabilities, and coefficients a clinician can inspect.

---

# Key takeaways (as the slides state them)

1. **Logistic regression** is the standard tool for binary classification.
2. **Naive Bayes** is useful when $p$ is very large (assumes feature independence within classes).
3. **KNN** is a non-parametric alternative; choose K via cross-validation.
4. **Always evaluate using a confusion matrix, ROC curve, and AUC — not just accuracy.**

---

# Formula sheet

| Concept | Formula |
|---|---|
| Logistic function | $p(X) = \dfrac{e^{\beta_0+\beta_1X}}{1+e^{\beta_0+\beta_1X}}$ |
| Odds | $\dfrac{p}{1-p} = e^{\beta_0+\beta_1X}$ |
| Logit (log-odds) | $\log\left(\dfrac{p}{1-p}\right) = \beta_0+\beta_1X$ |
| Odds ratio per unit | $e^{\beta_j}$ |
| Likelihood | $\prod_{i:y_i=1} p(x_i) \prod_{i:y_i=0}(1-p(x_i))$ |
| Softmax | $\Pr(Y=k|x) = \dfrac{e^{z_k}}{\sum_l e^{z_l}}$ |
| Naive Bayes | $\Pr(Y=k|x) \propto \pi_k \prod_{j=1}^{p} f_{kj}(x_j)$ |
| KNN probability | $\frac{1}{K}\sum_{i \in \mathcal{N}_0} I(y_i = j)$ |
| Recall / Sensitivity / TPR | $TP/(TP+FN)$ |
| Specificity / TNR | $TN/(TN+FP)$ |
| Precision | $TP/(TP+FP)$ |
| FPR (Type I) | $FP/(FP+TN)$ |
| FNR (Type II) | $FN/(FN+TP)$ |
| F1 | $2PR/(P+R)$ |
| Accuracy | $(TP+TN)/(TP+TN+FP+FN)$ |

---

# sklearn cheat sheet

| Task | Code |
|---|---|
| Logistic regression | `LogisticRegression(max_iter=1000)` |
| Unpenalised logistic | `LogisticRegression(penalty=None)` |
| Gaussian naive Bayes | `GaussianNB()` |
| KNN | `KNeighborsClassifier(n_neighbors=k)` |
| Hard predictions | `model.predict(X)` |
| **Probabilities** | `model.predict_proba(X)[:, 1]` |
| Custom threshold | `(proba >= 0.3).astype(int)` |
| Confusion matrix | `confusion_matrix(y_true, y_pred)` → `[[TN,FP],[FN,TP]]` |
| Plot it | `ConfusionMatrixDisplay.from_predictions(...)` |
| All metrics at once | `classification_report(y_true, y_pred)` |
| ROC curve | `roc_curve(y_true, y_prob)` → `fpr, tpr, thresholds` |
| AUC | `roc_auc_score(y_true, y_prob)` |
| Stratified split | `train_test_split(..., stratify=y)` |
| Stratified CV | `StratifiedKFold(5, shuffle=True, random_state=0)` |
| Log-odds → probability | `1 / (1 + np.exp(-z))` |
| Coefficients → odds ratios | `np.exp(model.coef_[0])` |

### Six mistakes that cost marks

1. **Using `predict()` when you need `predict_proba()`** — feeding hard labels to `roc_curve` gives a meaningless AUC.
2. **Taking `predict_proba()[:, 0]`** — that's P(negative class), which inverts your curve.
3. **Reading sklearn's confusion matrix in the slides' layout** — sklearn is actual-by-predicted, `[[TN,FP],[FN,TP]]`.
4. **Not scaling for KNN or regularised logistic regression** — costs 4+ points of accuracy here.
5. **Interpreting a logistic coefficient as a change in probability** — it's a change in *log-odds*.
6. **Reporting accuracy alone on imbalanced data** — 99% accuracy can mean catching zero positives.

---

# Quick self-test

Cover the answers.

1. **Why not linear regression for a binary Y?** — *It can predict below 0 and above 1, which aren't probabilities. It's not unreasonable as a classifier for two classes, but it fully breaks for K > 2 because numeric coding imposes a fake ordering.*
2. **β₁ = 0.0055. What does it mean?** — *Each unit of X adds 0.0055 to the log-odds, equivalently multiplies the odds by e^0.0055 ≈ 1.0055. It does NOT mean the probability rises by 0.0055.*
3. **Students default more often, yet the student coefficient is negative. How?** — *Confounding. Students carry higher balances, and balance drives default. Holding balance fixed, being a student is protective. Marginal and conditional relationships can have opposite signs.*
4. **When would you set the threshold to 0.1 instead of 0.5?** — *When a false negative costs far more than a false positive — cancer screening, fraud detection. 0.5 minimises overall error only when the two errors cost the same.*
5. **Model A: 99% accuracy. Model B: 91%. Which is better?** — *Not enough information. If 99% of cases are negative, model A may be predicting "negative" every time and catching nothing. Ask for the confusion matrix.*
6. **What does AUC = 0.85 literally mean?** — *Pick a random positive and a random negative case: there's an 85% chance the model scores the positive one higher.*
7. **Naive Bayes assumes independence, and your features correlate at 0.99. Should you abandon it?** — *Not necessarily. The probabilities become overconfident because evidence is double-counted, but the argmax often survives, so classification accuracy can stay fine. Don't trust the probabilities themselves.*
8. **KNN with K=1 vs K=n — which overfits?** — *K=1 overfits: maximally flexible, low bias, high variance. K=n underfits, predicting the majority class everywhere.*
9. **Your KNN has the best accuracy but the worst AUC. What's happening?** — *With small K, predict_proba takes only K+1 distinct values, so the model can't finely rank cases. It classifies well at 0.5 but has almost nothing to threshold on. Exactly what happens in this lab.*
10. **Which of the three classifiers need scaled features?** — *Logistic regression (because sklearn regularises by default) and KNN (distance-based). GaussianNB does not — it fits a mean and variance per feature per class, so per-feature affine rescaling cancels out.*
