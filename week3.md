# DSB102 — Week 3: Model Selection and Regularisation
### Complete study guide (lecture slides + solved lab)

---

## The one-paragraph version

Week 2 gave you a way to fit a model. Week 3 admits the problem with that: **the model that fits your training data best is almost never the model that predicts best.** Add predictors and training RSS falls forever, R² rises forever, and at some point you stop learning the signal and start memorising the noise. So this week is about *restraint* — three ways to impose it. **Forward stepwise** adds predictors one at a time and stops early. **Ridge** keeps everything but shrinks all coefficients. **Lasso** shrinks too, but pushes weak coefficients to exactly zero, deleting them. And because none of these can be tuned using training error, **cross-validation** becomes the referee for all of them.

> **Master metaphor — packing for a trip.** Training RSS is a friend who says "just bring it, you might need it." Follow that advice and you arrive with four suitcases you can't carry. The three methods are three packing strategies: **forward stepwise** picks items one at a time and stops when the bag is full; **ridge** brings everything but in travel-sized bottles; **lasso** brings full-sized bottles of the things you actually use and leaves the rest at home. **Cross-validation** is the airline scale that tells you, honestly, whether you've overpacked — because *you* can't be trusted to judge your own bag.

**The single idea underneath everything:** we accept a little **bias** (a deliberately worse fit on training data) in exchange for a large reduction in **variance** (less sensitivity to the particular sample you happened to collect). That trade is what "regularisation" means.

---

## Setup — imports used throughout

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import (LinearRegression, Ridge, Lasso,
                                  RidgeCV, LassoCV, ElasticNet)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
```

> ⚠️ **Terminology collision you must know.** The slides call the tuning parameter **λ (lambda)**. sklearn calls it **`alpha`**. They are the same thing. `Ridge(alpha=64.3)` *is* λ = 64.3.

---

# PART 1 — Why we need model selection at all

The linear model from Week 2 is unchanged: $Y = \beta_0 + \beta_1X_1 + \cdots + \beta_pX_p + \epsilon$

We keep it because linear models are **interpretable** and often **predict well**. We replace *ordinary least squares fitting* with something better for two reasons:

| Goal | Problem OLS has | Fix |
|---|---|---|
| **Prediction accuracy** | When $p$ is large — especially $p > n$ — OLS has huge variance, or no unique solution at all | Constrain the coefficients |
| **Model interpretability** | OLS gives every predictor a non-zero coefficient, including irrelevant ones | Force irrelevant coefficients to zero |

> **Why does $p > n$ break OLS entirely?** With more unknowns than equations there are infinitely many coefficient vectors that fit the training data *perfectly* (RSS = 0). OLS has no basis for choosing between them. **Metaphor:** asked to draw a line through one single point, you can draw infinitely many. You need an extra rule — "prefer the flattest one" — and that extra rule is exactly what a penalty term provides.

## 1.1 The two main approaches

**1. Subset Selection (Forward Stepwise)** — start with no predictors, add the most useful one at each step, fit by least squares on the chosen subset.

**2. Shrinkage / Regularisation** — fit with all $p$ predictors, but penalise large coefficients to reduce variance. Lasso can drive coefficients to exactly zero, giving variable selection for free.

> **Metaphor:** subset selection is a **light switch** — a predictor is in or out. Shrinkage is a **dimmer** — every predictor stays connected, but you turn the power down. Lasso is a dimmer *with a click-off at the bottom*, which is why it does both jobs at once.

## 1.2 Why not just search all possible models?

With $p$ predictors there are $2^p$ possible models:

| $p$ | Models |
|---|---|
| 10 | 1,024 |
| 30 | over 1 billion |
| 40 | over a trillion |

Two problems, and the second matters more than the first:

1. **Computationally infeasible** for large $p$.
2. **Searching a huge space inflates the chance of finding a model that looks good on training data purely by luck.**

> **Metaphor for problem 2:** hold a coin-flipping contest with a million entrants and someone will flip ten heads in a row. They are not a gifted flipper. Search enough models and one will fit your training noise beautifully — and it will be *worthless* on new data. The more models you try, the more suspicious you should be of the winner.

Forward stepwise is the practical alternative: it builds the model **greedily**, one predictor at a time, examining roughly $p^2/2$ models instead of $2^p$.

---

# PART 2 — Forward stepwise selection

**The algorithm, exactly as the slides state it:**

1. Let $M_0$ be the **null model** — no predictors, just the intercept. It predicts $\bar{y}$ for everyone.
2. For $k = 0, 1, \ldots, p-1$:
   - Consider all $p - k$ models that add **one** predictor to $M_k$.
   - Choose the best (smallest RSS / highest R²) among them; call it $M_{k+1}$.
3. Select a single best model from $M_0, \ldots, M_p$ using **cross-validated prediction error, $C_p$ (AIC), BIC, or Adjusted R²**.

> **Note the two-stage structure — this is the most commonly missed point.** Step 2 uses **training RSS** to decide *which* predictor joins next. Step 3 uses a **completely different criterion** to decide *when to stop*. Training RSS is fine for the first job (comparing models of the *same* size) and useless for the second (comparing models of *different* sizes).

**Not guaranteed to find the globally best subset**, but works well in practice.

> **Metaphor:** forward stepwise is a hiker who always steps in the steepest upward direction. Efficient, and usually gets somewhere high. But commit to a foothill early and you may never reach the true summit — a greedy algorithm cannot undo an earlier choice. This is why it can be **unstable with correlated predictors**: two near-identical predictors compete, one wins by a hair, and the other is then permanently redundant.

```python
lr = LinearRegression()

sfs = SequentialFeatureSelector(
    lr,
    n_features_to_select=5,
    direction="forward",      # "backward" for backward elimination
    cv=5,                     # scores each candidate by cross-validation
)
sfs.fit(X_tr, y_tr)

selected = [features[i] for i in sfs.get_support(indices=True)]
print(selected)

# get_support() gives a boolean mask for column selection
X_sel_tr = X_tr[:, sfs.get_support()]
X_sel_te = X_te[:, sfs.get_support()]
```

> ⚠️ **`SequentialFeatureSelector` will not choose `k` for you.** `n_features_to_select` is required — it does step 2 of the algorithm, not step 3. Choosing `k` is your job (Exercise 1 below). Passing `n_features_to_select="auto"` with a `tol` exists but behaves differently from the slide algorithm.

---

# PART 3 — Choosing the optimal model

**The core problem, stated by the slides:**

- The model with **all** predictors will always have the smallest RSS and largest R².
- We want low **test** error, not low **training** error.
- Training error is usually a **poor estimate** of test error.
- Therefore **RSS and R² are not suitable** for comparing models with different numbers of predictors.

> **Metaphor:** judging a student by their performance on the exact questions they revised. Everyone scores 100%. The score tells you nothing about who understands the material — you need *unseen* questions.

## 3.1 The four adjustment criteria

All four adjust training error for model size, so models of different sizes become comparable.

| Criterion | Direction | Character |
|---|---|---|
| **Mallow's $C_p$** | Lower better | Penalises training RSS for each extra predictor |
| **AIC** | Lower better | Similar penalty; works across many model types, not just linear |
| **BIC** | Lower better | **Heavier** penalty than AIC → selects **smaller, simpler** models |
| **Adjusted R²** | **Higher** better | Unlike plain R², *decreases* if a predictor adds noise rather than signal |

**In plain English (the slides' own phrasing):** all four ask *"Does adding this predictor actually help, or am I just memorising the training data?"*

> **Metaphor:** four examiners marking the same essay. All reward good content and deduct for padding — they only disagree on how harshly to deduct. **BIC is the strictest marker** (its penalty grows with $\log n$, so on large datasets it becomes very demanding), **AIC is more lenient**. In the slides' Credit figure this shows up exactly as you'd expect: **BIC picks 4 predictors, $C_p$ picks 6, Adjusted R² picks 7.** Same data, same models, three different answers — the difference is purely the size of the fine.

⚠️ **Adjusted R² is the odd one out — higher is better.** Mixing up the direction is a classic exam error. Memorise: *three lowers and one higher.*

```python
# Adjusted R2 and AIC/BIC come free from statsmodels (Week 2 tooling)
import statsmodels.api as sm
m = sm.OLS(y, sm.add_constant(X)).fit()
print(m.rsquared_adj, m.aic, m.bic)

# Adjusted R2 from the definition, to see the penalty explicitly:
n, p = len(y), X.shape[1]
adj = 1 - (1 - m.rsquared) * (n - 1) / (n - p - 1)
#                            ^^^^^^^^^^^^^^^^^^^^ grows as p grows -> drags R2 down
```

---

# PART 4 — Why one validation split isn't enough

The slides list three drawbacks of the simple train/validation split:

1. The estimate is **highly variable** — it depends on precisely which observations landed in which half.
2. Only a **subset** of observations is used to fit the model.
3. Consequently the validation error tends to **overestimate** the true test error of a model fit on the full dataset.

**Slide 9 asks "Why?" and leaves it hanging. The answer:** more training data almost always produces a better model. A model trained on 50% of your data is genuinely worse than one trained on 100%. So the error you measure is the error of a *handicapped* model — pessimistic about the model you'd actually deploy.

> **Metaphor:** judging a runner's marathon time from a race they ran with a weighted vest on. The time is real, but it's not the time they'd run without it. Cross-validation reduces the weight of the vest — with 5 folds each model trains on 80% instead of 50%, so the handicap is far smaller.

You saw drawback 1 numerically in Week 2: on one split the News model scored better, over 30 splits it scored worse. **One split is a coin flip.**

---

# PART 5 — K-fold cross-validation

**The procedure:**

1. Randomly divide the data into $K$ equal-sized parts (folds).
2. Leave out part $k$; fit the model on the other $K-1$ parts combined.
3. Predict the left-out part and record the error.
4. Repeat for every $k = 1, \ldots, K$ and average the $K$ errors.

> **Metaphor:** a study group of five where everyone takes a turn writing the practice exam while the other four sit it. Nobody grades their own paper, everybody eventually gets tested, and no single unlucky exam decides the outcome. Averaging is what kills the coin-flip variance.

**Advantages the slides list:**
- Direct estimate of test error
- Doesn't require estimating the error variance $\sigma^2$ (which $C_p$ and AIC *do* need)
- Applies to a far wider range of model selection problems

That second point is quietly important. $C_p$, AIC and BIC are *theoretical approximations* to test error that assume you know $\sigma^2$ and that the model is roughly correct. **CV just measures test error directly and assumes almost nothing.** That's why CV became the default.

```python
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# sklearn maximises scores, so MSE is returned NEGATED
scores = cross_val_score(LinearRegression(), X_tr, y_tr,
                         cv=kf, scoring="neg_mean_squared_error")

cv_rmse = np.sqrt(-scores.mean())    # note the minus sign
print(f"CV RMSE = {cv_rmse:.4f}  (per-fold sd {np.sqrt(-scores).std():.4f})")
```

> ⚠️ **The negative-score trap.** `cross_val_score` returns `neg_mean_squared_error` — already negated so that "higher is better". Forget the minus sign and `np.sqrt` hands you a `nan`. Also: **always pass `shuffle=True`** if your data has any ordering to it, or your folds may be systematically different from each other.

**Choosing K:** K = 5 or 10 are standard. K = n is leave-one-out (LOOCV) — nearly unbiased, but high-variance and slow. K = 5 is the usual compromise, and it's what the lab uses throughout.

---

# PART 6 — Ridge regression

OLS minimises RSS alone. Ridge adds a penalty:

$$\underset{\beta}{\text{minimise}} \quad \underbrace{\text{RSS}}_{\text{fit}} \; + \; \underbrace{\lambda \sum_{j=1}^{p} \beta_j^2}_{\text{penalty}}$$

- $\lambda \geq 0$ controls the trade-off. **Larger λ = more shrinkage.**
- Coefficients shrink toward zero but **never reach exactly zero**.

**In plain English (the slides):** *"Find the best fit, but don't let any coefficient get too large."* Ridge keeps all predictors but makes them smaller and more stable.

**The two endpoints are worth memorising:**

| λ | Result |
|---|---|
| $\lambda = 0$ | Penalty vanishes → **exactly OLS** |
| $\lambda \to \infty$ | All coefficients crushed to ~0 → **the null model**, predicting $\bar{y}$ |

Everything useful lives between them, and CV finds where.

> **Metaphor:** λ is a **budget on total coefficient size**. With a generous budget you spend freely and chase every wiggle in the training data. With a tight budget you must spend only on predictors that really earn it. **The magic is that a slightly-too-tight budget beats a generous one on new data** — because much of what a generous budget buys is noise.

> **Why shrinkage helps with collinearity specifically:** in Week 2 you saw that collinear predictors produce wildly unstable coefficients — one huge positive paired with one huge negative, cancelling out. Those pairs have *enormous* $\sum \beta_j^2$, so ridge's penalty finds them intolerable and forces them to something sensible and stable. **Ridge is the direct cure for the disease VIF diagnoses.**

```python
ridge = Ridge(alpha=1.0).fit(X_tr, y_tr)     # alpha IS lambda

# Coefficient path — how every coefficient evolves as lambda grows
alphas = np.logspace(-3, 4, 100)
coefs = np.array([Ridge(alpha=a).fit(X_tr, y_tr).coef_ for a in alphas])

fig, ax = plt.subplots(figsize=(9, 5))
for j in range(coefs.shape[1]):
    ax.plot(np.log10(alphas), coefs[:, j], lw=1)
ax.axhline(0, color="k", lw=0.7, ls="--")
ax.set(xlabel="log10(lambda)", ylabel="Coefficient",
       title="Ridge coefficient paths")
plt.show()
```

**How to read a coefficient path plot:** the far left is OLS (λ→0, coefficients wild and spread out). Moving right, every line is squeezed toward the dashed zero line. In **Ridge**, lines approach zero asymptotically but never touch it.

---

# PART 7 — Standardisation (the step you cannot skip)

The slides devote a whole slide to this, and it's the most common practical mistake in the whole topic.

**OLS is scale equivariant.** Multiply $X_j$ by a constant $c$ and $\hat{\beta}_j$ simply divides by $c$ — the product $X_j\hat{\beta}_j$, and therefore every prediction, is unchanged. Measure height in metres or centimetres; OLS does not care.

**Ridge and Lasso are NOT.** The penalty $\sum \beta_j^2$ sums raw coefficient values. A predictor measured in small units gets a large coefficient, so it is punished harder — for a reason that has nothing to do with its usefulness.

> **Metaphor:** a fine levied on the *number printed on the price tag*, ignoring the currency. An item priced ¥5000 gets fined ten times harder than one priced £50, even if they cost the same. Standardising converts every predictor to the same currency before the fine is calculated.

**Always standardise before fitting ridge or lasso**: subtract the mean, divide by the standard deviation.

```python
# Simple version (what the lab does)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# BETTER: a pipeline, which re-fits the scaler inside every CV fold
pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 100), cv=5))
pipe.fit(X_train_raw, y_train)
print(pipe[-1].alpha_)
```

> **Why the pipeline version is more correct.** Calling `scaler.fit_transform` on the *whole* dataset before splitting lets the test set's mean and standard deviation influence the training data — a mild form of **data leakage**. A pipeline re-learns the scaling inside each fold, using training rows only.
>
> **Honest scale of the problem here:** I ran both on the lab's data. Lab approach → test RMSE **0.6169**; leak-free pipeline → **0.6173**. The difference is negligible on this dataset. But it's a habit worth forming, because on smaller datasets or with heavier preprocessing (imputation, PCA, target encoding) the same mistake can be seriously misleading.

**Note:** the intercept is never penalised, by any of these methods. Shrinking it would just bias all predictions toward zero for no benefit. sklearn handles this for you.

---

# PART 8 — The Lasso

Identical structure, one change — absolute values instead of squares:

$$\underset{\beta}{\text{minimise}} \quad \underbrace{\text{RSS}}_{\text{fit}} \; + \; \underbrace{\lambda \sum_{j=1}^{p} |\beta_j|}_{\text{penalty}}$$

**This small change has a big effect:** when λ is large enough, some coefficients are pushed to **exactly zero**. Lasso therefore performs **automatic variable selection** — irrelevant predictors leave the model entirely.

**In plain English (the slides):** *"Find the best fit, and feel free to drop predictors that aren't pulling their weight."*

## 8.1 Why does L1 zero things out but L2 doesn't?

This is the single most-asked question of the week, and it deserves a real answer.

**The gradient argument (most intuitive).** Think about the force pulling a coefficient toward zero as it gets small:

- **Ridge penalty** $\beta^2$ → derivative $2\beta$. As $\beta \to 0$, the pulling force $\to 0$. **The penalty gives up right when it's about to finish the job.**
- **Lasso penalty** $|\beta|$ → derivative $\pm 1$. Constant. **The force never weakens**, so it pushes the coefficient right through zero and pins it there.

**The geometric argument (what the slides' figures show).** The penalty defines a constraint region: a **circle** for ridge ($\beta_1^2 + \beta_2^2 \le s$), a **diamond** for lasso ($|\beta_1| + |\beta_2| \le s$). The solution sits where the RSS contours first touch that region. A **diamond has sharp corners, and its corners lie on the axes** — where a coefficient is exactly zero. A contour is very likely to hit a corner. A **circle has no corners**, so a tangent point with a coordinate of exactly zero is a measure-zero coincidence.

> **Metaphor:** ridge is a **round pond**, lasso is a **diamond-shaped pond with four sharp points**. Throw a ring at each. On the round pond it can land anywhere on the rim. On the diamond it catches on a point — and the points are exactly the places where one variable has been switched off.

```python
lasso = Lasso(alpha=0.01, max_iter=10000).fit(X_tr, y_tr)

coef = pd.Series(lasso.coef_, index=features)
print(f"Zeroed: {(coef == 0).sum()} / {len(features)}")
print(coef[coef != 0].sort_values(key=abs, ascending=False).round(4))
```

> ⚠️ **`max_iter=10000`.** Lasso has no closed-form solution — it's solved iteratively by coordinate descent. The default `max_iter=1000` frequently throws a `ConvergenceWarning` on small λ. Raise it; the cost is trivial.

---

# PART 9 — Ridge vs Lasso: when to use which

**The slides are careful here, and you should be too: neither universally dominates.**

- **Lasso may perform better when the response depends on relatively few predictors** (a *sparse* signal).
- Ridge may perform better when many predictors each contribute a little (a *dense* signal).
- **The number of relevant predictors is never known in advance for real data.**
- So: **use cross-validation to decide which approach suits your dataset.**

| | Interpretability | Stability with correlated predictors | Variable selection |
|---|---|---|---|
| **Forward stepwise** | High — discrete in/out | **Poor** — greedy, unstable | Yes |
| **Ridge** | Lower — keeps all $p$ | **Excellent** | No |
| **Lasso** | High — sparse | Moderate (picks one of a correlated group arbitrarily) | Yes |

**When to use each (slide 21):**

- **Forward stepwise** — you need an explicitly interpretable reduced model; moderate $p$ (tens of predictors).
- **Ridge** — many predictors, all expected to contribute something; **high multicollinearity**.
- **Lasso** — sparse signal, only a few predictors truly matter; you want automatic selection.

**In all cases: choose λ (or the number of steps) by cross-validation.**

> **Worth knowing though it's not in the slides:** `ElasticNet` blends both penalties, which handles correlated groups better than lasso alone — lasso tends to arbitrarily pick one member of a correlated group and zero the rest, whereas elastic net keeps or drops them together.

---

# PART 10 — Selecting the tuning parameter λ

**The recipe, exactly as the slides give it:**

1. Choose a grid of λ values.
2. Compute the cross-validation error for each λ.
3. Select the λ with the smallest CV error.
4. **Refit the model using all available observations at the selected λ.**

**Step 4 is the one people forget.** CV's job is to *choose* λ, not to produce the final model. Once λ is chosen, throw away the fold models and refit on everything — more data, better model. `RidgeCV` and `LassoCV` do this automatically.

```python
alphas = np.logspace(-3, 4, 100)      # ALWAYS search on a log scale

ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(X_tr, y_tr)
print(f"Best lambda: {ridge_cv.alpha_:.4f}")

lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5,
                   max_iter=10000).fit(X_tr, y_tr)
print(f"Best lambda: {lasso_cv.alpha_:.4f}")

# Both are already refitted on the full training set at the chosen lambda —
# step 4 is done for you. Just call .predict().
```

> **Why `np.logspace` and not `np.linspace`?** λ matters *multiplicatively*. The gap between 0.001 and 0.01 is as meaningful as the gap between 100 and 1000. A linear grid from 0 to 100 would waste nearly every point in a region where nothing changes and skip straight past the interesting range.

> ⚠️ **Check your chosen λ isn't at the edge of your grid.** If `alpha_` comes back as your largest or smallest candidate, the true optimum is probably outside the range you searched — widen it and re-run.

---

# PART 11 — The slides' three worked examples

## 11.1 Forward stepwise — training RSS can't tell you when to stop

| Step | Predictors | Training RSS |
|---|---|---|
| $M_0$ | none | 100 |
| $M_1$ | X2 | 40 |
| $M_2$ | X2, X1 | 25 |
| $M_3$ | X2, X1, X3 | 22 |

**Key insight:** training RSS **always** falls as predictors are added. It is monotonic. It therefore contains **zero information** about when to stop.

## 11.2 Cross-validation — the criterion that *does* turn around

| Model | Training RSS | 5-fold CV Error |
|---|---|---|
| $M_0$ | 100 | 24 |
| $M_1$ | 40 | 15 |
| $M_2$ | 25 | **11 (lowest)** |
| $M_3$ | 22 | 13 |

**Key insight:** RSS keeps falling to $M_3$, but CV error **turns back up after $M_2$**. Adding X3 buys 3 units of training fit and costs 2 units of genuine predictive accuracy — X3 is fitting noise. **CV selects $M_2$.**

> **This U-shape is the single most important picture in the course.** Its left arm is **underfitting** (too rigid, high bias); its right arm is **overfitting** (too flexible, high variance). Every tuning parameter you meet from here on — λ, tree depth, k in kNN, number of layers — is a dial along this same curve, and the job is always to find the bottom.

## 11.3 Ridge vs Lasso side by side

Two predictors. OLS gives $\hat{\beta}_1 = 3$ (strong signal) and $\hat{\beta}_2 = 0.1$ (weak). Apply λ = 2:

| | OLS | Ridge (λ=2) | Lasso (λ=2) |
|---|---|---|---|
| $\hat{\beta}_1$ | 3.0 | 1.0 | 1.0 |
| $\hat{\beta}_2$ | 0.1 | 0.03 | **0 (excluded)** |

**Key insight:** *Ridge keeps everything but makes it smaller. Lasso keeps only what matters.*

Notice what happened to $\beta_2$ under ridge: 0.1 → 0.03. Shrunk by 70%, still not zero, and now a small distracting number in your output that you have to explain to someone. Lasso just removed it.

---

# PART 12 — The lab, worked through

**Dataset:** `Hitters` — baseball players' career statistics predicting salary. 263 players after dropping missing salaries, **18 predictors**, split 210 train / 53 test.

**Two preprocessing choices worth understanding:**

```python
df = df.dropna(subset=["Salary"])
df["LogSalary"] = np.log(df["Salary"])        # (1) log-transform the response
df = pd.get_dummies(df, columns=["League", "Division", "NewLeague"],
                    drop_first=True, dtype=float)   # (2) K-1 dummies (Week 2!)
df = df.drop(columns=["Salary"])
```

1. **Why log the salary?** Salaries are right-skewed with a long tail of superstars, which produces the fan-shaped residuals you diagnosed in Week 2. Logging compresses that tail. This is the Week 2 heteroscedasticity remedy applied before you even start.
2. **`drop_first=True`** is the K−1 dummy rule from Week 2, avoiding the dummy variable trap.

## Part 1 — Forward stepwise

**Tutorial, k = 5:** selected `['Hits', 'Walks', 'Years', 'CHits', 'CWalks']` → **test RMSE 0.6674**

**Exercise 1 — CV RMSE across k:**

| k | 3 | 5 | **8** | 12 | 16 |
|---|---|---|---|---|---|
| CV RMSE | 0.6424 | 0.6355 | **0.6319** | 0.6354 | 0.6592 |

**Best k = 8**, CV RMSE 0.6319. And there's the **U-shape from slide 23**, in real data — error falls to k=8, then climbs again.

**Exercise 1, task 4 — "what is the risk of maximising training R²?"** You'd pick k = 18 every time, because R² cannot decrease. You would select the most overfit model available, guaranteed, by construction.

## Part 2 — Ridge

**Best λ = 64.28** → **test RMSE 0.6169**. Plain OLS → **0.6539**.

**Exercise 2 answers:**
1. OLS test RMSE 0.6539.
2. **Ridge wins**, by about 5.7%.
3. **No coefficient reaches exactly zero** — the L2 gradient $2\beta$ vanishes as $\beta \to 0$, so the shrinking force dies out before finishing the job.
4. Ridge is preferred when predictors are numerous, correlated, and each contributes something.

## Part 3 — Lasso

**Best λ = 0.0266**, **9 of 18 coefficients zeroed**, **test RMSE 0.6326**.

Surviving predictors:

| Predictor | Coefficient |
|---|---|
| Hits | 0.2612 |
| Years | 0.2431 |
| CHits | 0.2124 |
| Walks | 0.0925 |
| Division_W | −0.0694 |
| HmRun | 0.0319 |
| PutOuts | 0.0263 |
| Errors | −0.0207 |
| League_N | 0.0149 |

**Exercise 3, task 2 — do they match forward stepwise?** Forward-5 chose Hits, Walks, Years, CHits, CWalks. Lasso keeps **four of those five** but drops `CWalks` — while adding HmRun, PutOuts, Errors, Division_W, League_N. **Substantial agreement on the strong signals, disagreement at the margin.** That's the honest picture: different methods broadly agree on what matters most and diverge on borderline predictors.

⚠️ **λ = 64.28 for ridge vs λ = 0.0266 for lasso.** These are **not comparable numbers.** One penalty sums squares, the other sums absolute values, so the scales are entirely different. Never compare a ridge λ to a lasso λ.

## Part 4 — Final comparison

| Method | Test RMSE |
|---|---|
| **Ridge** | **0.6169** 🏆 |
| Lasso | 0.6326 |
| OLS (all 18) | 0.6539 |
| Forward stepwise (8 feats) | 0.6564 |

**Exercise 4, task 2 — which wins, and is the improvement large?** Ridge wins. The gain over OLS is ~5.7% RMSE — **real but modest.** Regularisation is usually a refinement, not a transformation.

**Exercise 4, task 3 — why compare on the test set rather than CV RMSE?** Because CV RMSE was **used to choose** λ and k. A number you optimised against is no longer an unbiased estimate of anything — you've fitted to it. The test set is the only data untouched by any decision, so it's the only honest referee.

You can see this bias in the numbers: forward stepwise had **CV RMSE 0.6319** but **test RMSE 0.6564**. CV was optimistic by 4%, precisely because k was picked to minimise it.

### 🔍 Why did Ridge win? (the lab doesn't say, but the data does)

The `Hitters` predictors include six career totals — `CAtBat`, `CHits`, `CRuns`, `CRBI`, `CWalks`, `CHmRun` — that are **near-duplicates of each other**:

```python
career = ["CAtBat", "CHits", "CRuns", "CRBI", "CWalks", "CHmRun"]
print(df[career].corr().round(3))
```

Correlations run from **0.79 to 0.995**. Running the Week 2 VIF check on them:

| | CAtBat | CHits | CRuns | CRBI | CWalks | CHmRun |
|---|---|---|---|---|---|---|
| **VIF** | 141.5 | **337.3** | 104.0 | 105.5 | 11.3 | 34.7 |

Every one is far past the "serious problem" threshold of 10. A player with many career at-bats has, necessarily, many career hits and runs. **This is a textbook high-multicollinearity dataset — exactly the case slide 21 says Ridge is for, and Ridge duly wins.** Lasso must arbitrarily pick one of six near-identical variables and discard the rest, losing information; ridge spreads the credit smoothly across all of them.

**This is the moment Week 2 and Week 3 connect:** VIF *diagnosed* the disease, ridge *treats* it.

### 🔍 Why did forward stepwise finish last?

Same reason, from the other direction. Greedy selection among six near-identical career stats is close to a coin flip — whichever wins by a hair blocks the others permanently. Slide 20's warning that stepwise "can be unstable (correlated predictors)" is not theoretical here; it's the reason it placed fourth.

---

# Key takeaways (as the slides state them)

1. **Forward Stepwise Selection** — greedy; adds the most useful predictor at each step. Use CV or BIC to choose how many predictors to include.
2. **Ridge Regression (ℓ2 penalty)** — shrinks all coefficients toward zero (never exactly zero); best when many predictors each contribute a small amount.
3. **Lasso (ℓ1 penalty)** — sets some coefficients to exactly zero, giving automatic variable selection; best when only a few predictors truly matter.
4. **Always standardise predictors** before fitting ridge or lasso.
5. **Choose λ via cross-validation** (`RidgeCV` / `LassoCV` in sklearn).

---

# Formula sheet

| Concept | Formula |
|---|---|
| OLS objective | $\min \; \text{RSS}$ |
| Ridge objective | $\min \; \text{RSS} + \lambda \sum_{j=1}^{p} \beta_j^2$ |
| Lasso objective | $\min \; \text{RSS} + \lambda \sum_{j=1}^{p} \|\beta_j\|$ |
| Elastic net | $\min \; \text{RSS} + \lambda_1 \sum \|\beta_j\| + \lambda_2 \sum \beta_j^2$ |
| Number of subsets | $2^p$ |
| Models forward stepwise fits | $1 + p(p+1)/2$ |
| Adjusted R² | $1 - \dfrac{(1-R^2)(n-1)}{n-p-1}$ |
| K-fold CV error | $\frac{1}{K}\sum_{k=1}^{K} \text{MSE}_k$ |
| Standardisation | $z_j = (x_j - \bar{x}_j) / s_j$ |

---

# sklearn cheat sheet

| Task | Code |
|---|---|
| Standardise | `StandardScaler().fit_transform(X)` |
| Forward stepwise | `SequentialFeatureSelector(lr, n_features_to_select=k, direction="forward", cv=5)` |
| Which features chosen | `sfs.get_support()` (mask) / `get_support(indices=True)` |
| Ridge at fixed λ | `Ridge(alpha=1.0)` |
| Lasso at fixed λ | `Lasso(alpha=0.01, max_iter=10000)` |
| Ridge with CV-tuned λ | `RidgeCV(alphas=np.logspace(-3, 4, 100), cv=5)` |
| Lasso with CV-tuned λ | `LassoCV(alphas=..., cv=5, max_iter=10000)` |
| Chosen λ | `model.alpha_` |
| Coefficients | `model.coef_` / intercept `model.intercept_` |
| K-fold CV score | `cross_val_score(est, X, y, cv=5, scoring="neg_mean_squared_error")` |
| Leak-free preprocessing | `make_pipeline(StandardScaler(), RidgeCV(...))` |

### Five mistakes that cost marks

1. **Not standardising before ridge/lasso.** No error is raised; the answer is just wrong.
2. **Confusing λ with `alpha`.** Same thing, different name.
3. **Forgetting the minus in `neg_mean_squared_error`.** `np.sqrt` of a negative gives `nan`.
4. **Reporting CV error as your final performance figure.** It's biased once you've tuned against it. Report test error.
5. **Comparing a ridge λ to a lasso λ.** Different penalty scales entirely.

---

# Quick self-test

Cover the answers.

1. **Why can't training RSS choose the number of predictors?** — *It decreases monotonically, so it always picks the largest model. It's not a criterion, it's a ratchet.*
2. **Ridge with λ = 0 gives you what?** — *Exactly OLS. And λ → ∞ gives the null model predicting ȳ.*
3. **Why does lasso zero coefficients but ridge doesn't?** — *The |β| derivative is a constant ±1, so the shrinking force never weakens near zero. The β² derivative is 2β, which vanishes as β→0. Geometrically: a diamond has corners on the axes; a circle doesn't.*
4. **You forgot to standardise before ridge. What happens?** — *Predictors measured in small units get large coefficients and are penalised harder for no substantive reason. The model silently produces the wrong answer.*
5. **BIC picks 4 predictors, adjusted R² picks 7. Who's wrong?** — *Neither. BIC has a heavier complexity penalty, so it always prefers smaller models. They're answering slightly different questions.*
6. **Your CV RMSE is 0.63 but test RMSE is 0.66. Is something broken?** — *No — expected. You chose the model to minimise CV error, so CV error is optimistically biased. That gap is why the test set is held back.*
7. **p = 500, n = 100. Can you fit OLS?** — *Not uniquely — infinitely many coefficient vectors give RSS = 0. Ridge or lasso resolve it by adding a preference for small coefficients.*
8. **Six predictors are correlated at r > 0.95. Ridge or lasso?** — *Ridge. Lasso will arbitrarily keep one and drop five, throwing away information; ridge spreads credit across the group. This is exactly the Hitters career-stats situation.*
