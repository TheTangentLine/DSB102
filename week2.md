# DSB102 — Week 2: Linear Regression
### Complete study guide (lecture slides + solved lab)

---

## The one-paragraph version

Linear regression draws a straight line (or flat plane) through a cloud of data and uses it to predict. You choose the line by making the **total squared miss** as small as possible. Then you spend the rest of the week asking three questions about that line: *Is it real?* (t-tests, F-test, p-values), *Is it any good?* (R², RSE, test RMSE), and *Is it trustworthy?* (residual plots, VIF). Everything in Week 2 is one of those three questions.

> **Master metaphor:** You're laying a **straight steel ruler** over a winding country road. The road is the truth. The ruler is your model. It will never trace every bend — but if you position it well, it tells you roughly where the road goes, and it's cheap, portable, and easy to read. The whole week is about *how to position the ruler* and *how to know when the road is too bendy for a ruler at all*.

---

## Setup — imports used throughout

Every snippet below assumes these are already run.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

**The one pattern to memorise.** Almost everything else is a variation on it:

```python
X = sm.add_constant(df[["TV", "Radio"]])    # statsmodels does NOT add an intercept
model = sm.OLS(df["Sales"], X).fit()        # note the order: y FIRST, then X
print(model.summary())
```

> ⚠️ **The classic silent failure.** sklearn adds an intercept for you; statsmodels does not. Forget `add_constant` and you force the line through the origin — the model still fits, still prints a tidy table, and is quietly wrong.

---

# PART 1 — The model itself

## 1.1 Simple linear regression

$$Y = \beta_0 + \beta_1 X + \epsilon$$

| Symbol | Name | Plain English |
|---|---|---|
| $\beta_0$ | Intercept | Where the line crosses the y-axis — the predicted Y when X = 0 |
| $\beta_1$ | Slope | How much Y moves per 1-unit increase in X |
| $\epsilon$ | Error term | Everything the line can't explain: randomness, missing variables, noise |

Once fitted, we get **estimates** with hats on them and predict:

$$\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x$$

> **The hats matter.** $\beta_1$ is the true slope of the universe — you will never see it. $\hat{\beta}_1$ is your best guess from your one sample. It's the difference between *a person's actual height* and *your estimate of it from a blurry photo*. Most of the statistics this week exists to say **how blurry the photo is**.

## 1.2 Estimation by least squares

For each data point:
- **Residual:** $e_i = y_i - \hat{y}_i$ — the vertical gap between the actual dot and the line.
- **RSS (residual sum of squares):** $\text{RSS} = e_1^2 + e_2^2 + \cdots + e_n^2$

**Least squares** = choose $\hat{\beta}_0, \hat{\beta}_1$ to make RSS as small as possible.

> **Metaphor:** Imagine a rigid rod, and a rubber band stretching vertically from every data point to the rod. Let go. The rod settles into the position where the total tension is lowest. That resting position *is* the least-squares line. Squaring is what makes rubber bands behave like rubber bands — pull twice as far, and the punishment is four times worse. This is why **outliers dominate** a regression: one point ten units away hurts as much as a hundred points one unit away.

Why square instead of taking absolute values?
1. Big errors get punished disproportionately (usually desirable).
2. It gives a smooth, single-minimum problem with a clean closed-form solution.
3. Positive and negative errors can't cancel out.

**The slides are explicit:** you don't need to derive the estimates by hand — Python/R does it. But for reference, in the simple case:

$$\hat{\beta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \qquad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

Note the second formula: **the line always passes through $(\bar{x}, \bar{y})$** — the centre of mass of the data. The line pivots around the average point.

```python
# Fit a simple regression
X_tv = sm.add_constant(df["TV"])
model_tv = sm.OLS(df["Sales"], X_tv).fit()

print(model_tv.summary().tables[1])       # tables[1] = just the coefficient table

# Verify the formulas by hand — no magic involved
x, y = df["TV"], df["Sales"]
b1 = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean())**2)
b0 = y.mean() - b1 * x.mean()
print(b0, b1)                             # matches model_tv.params exactly

# The line passes through the centre of mass:
print(b0 + b1 * x.mean(), y.mean())       # identical
```

**Plotting the fit:**

```python
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(df["TV"], df["Sales"], alpha=0.4, s=20, color="steelblue")
xs = np.linspace(0, 300, 200)
ax.plot(xs, model_tv.params.iloc[0] + model_tv.params.iloc[1] * xs, "r-", lw=2)
ax.set(xlabel="TV spend", ylabel="Sales", title="Sales ~ TV")
plt.show()
```

## 1.3 The five advertising questions (the framing device for the whole week)

The Advertising dataset (TV, Radio, Newspaper → Sales) is used to motivate everything:

1. Is there a relationship between advertising budget and sales? → **F-test / t-test**
2. How strong is it? → **R²**
3. How accurately can we predict? → **RSE, prediction intervals, test RMSE**
4. Is the relationship linear? → **residual plots**
5. Is there synergy between media? → **interaction terms**

If you can name the tool for each question, you understand Week 2.

---

# PART 2 — Is the relationship real? (Inference)

## 2.1 Standard error (SE)

The SE of an estimate says **how much it would bounce around if you repeated the whole study with a fresh sample**. Smaller SE = more reliable estimate.

> **Metaphor:** You weigh yourself once and get 72.4 kg. The SE is the answer to "if I stepped off and back on 100 times, how much would the reading jump around?" A scale with SE = 0.1 kg is trustworthy. A scale with SE = 8 kg is telling you almost nothing, even though it produced a confident-looking number to one decimal place.

**SE shrinks when:** you have more data (n ↑), the noise is small (σ ↓), or your X values are spread out. That last one is underappreciated — it's much easier to judge a line's slope from points spread across the room than from points bunched in one corner.

## 2.2 Residual standard error (RSE)

Because true error variance σ² is unknown, we estimate it:

$$\text{RSE} = \sqrt{\frac{\text{RSS}}{n-2}}$$

This is **the typical size of a miss**, in the same units as Y.

> **Metaphor:** RSE is your average distance from the bullseye, in centimetres. If you're predicting sales in thousands of units and RSE = 3.04, then your typical prediction is off by about 3,040 units — regardless of how impressive the R² looks. **RSE is the one that answers "is this useful in practice?"**

Why $n-2$? You spent two degrees of freedom estimating $\hat{\beta}_0$ and $\hat{\beta}_1$. In multiple regression it becomes $n - p - 1$.

```python
# Standard errors of each coefficient
print(model_tv.bse)

# RSE — note there is NO built-in .rse attribute; you compute it yourself
rse = np.sqrt(model_tv.mse_resid)         # mse_resid = RSS / (n - p - 1)
print(f"RSE = {rse:.2f}")

# By hand, to see where it comes from:
rss = model_tv.ssr                        # residual sum of squares
n = len(df)
print(np.sqrt(rss / (n - 2)))             # same number
```

## 2.3 Confidence intervals

$$\hat{\beta}_1 \pm 2 \cdot \text{SE}(\hat{\beta}_1)$$

Approximately 95% of intervals built this way, across repeated samples, will contain the true $\beta_1$.

> **Careful metaphor — this is the classic exam trap.** The interval is the **net**, not the fish. The true $\beta_1$ is a fixed number sitting still. *Your net* is what's random — it lands in a slightly different place each sample. "95% confidence" means the net-casting *procedure* catches the fish 95% of the time. It does **not** mean "there's a 95% probability that $\beta_1$ is in this particular interval" — $\beta_1$ is either in it or it isn't.

**Shortcut:** if the 95% CI does **not** contain 0, then the coefficient is significant at the 5% level. CI and t-test always agree.

```python
print(model_tv.conf_int())                # 95% by default
print(model_tv.conf_int(alpha=0.01))      # 99% intervals — wider

# The rule-of-thumb version from the slides (uses 2 instead of the exact t):
b1, se = model_tv.params.iloc[1], model_tv.bse.iloc[1]
print(b1 - 2*se, b1 + 2*se)

# Does the interval exclude zero? Then it's significant at 5%.
lo, hi = model_tv.conf_int().iloc[1]
print("significant" if lo > 0 or hi < 0 else "not significant")
```

## 2.4 Hypothesis testing

$$H_0: \beta_1 = 0 \quad \text{(no relationship)} \qquad H_A: \beta_1 \neq 0 \quad \text{(some relationship)}$$

If $\beta_1 = 0$, the model collapses to $Y = \beta_0 + \epsilon$ and X is irrelevant.

$$t = \frac{\hat{\beta}_1 - 0}{\text{SE}(\hat{\beta}_1)} \sim t_{n-2}$$

The t-statistic counts **how many standard errors your estimate sits away from zero**.

- **p-value** = probability of seeing a |t| this large *if $H_0$ were true*.
- **Small p (< 0.05)** → what you saw would be unlikely under "no relationship" → reject $H_0$.
- **Large p** → no evidence against $H_0$. (Note: *no evidence against*, **not** "proof there's no effect.")

> **Metaphor:** The p-value is a **surprise meter**. You assume the world is boring (no effect), then ask: "how shocking is my data under that assumption?" A p-value of 0.001 means "this would happen 1 time in 1000 by pure luck" — so the boring assumption is probably wrong. A p-value of 0.24 means "yeah, that happens all the time by chance" — shrug.
>
> **What p is NOT:** it is not the probability that $H_0$ is true, and it is not a measure of effect size. A tiny, useless effect will produce a microscopic p-value if n is large enough. **Significance ≠ importance.**

```python
print(model_tv.tvalues)                   # t-statistics
print(model_tv.pvalues)                   # two-sided p-values

# Reconstruct the t-stat and p-value manually:
from scipy import stats
t = model_tv.params.iloc[1] / model_tv.bse.iloc[1]
p = 2 * (1 - stats.t.cdf(abs(t), df=len(df) - 2))    # n - 2 degrees of freedom
print(t, p)

# Which predictors survive at the 5% level?
print(model_tv.pvalues[model_tv.pvalues < 0.05])
```

---

# PART 3 — How good is the model? (Model fit)

## 3.1 R²

$$R^2 = \frac{\text{TSS} - \text{RSS}}{\text{TSS}} = 1 - \frac{\text{RSS}}{\text{TSS}}, \qquad \text{TSS} = \sum(y_i - \bar{y})^2$$

| Quantity | Meaning |
|---|---|
| **TSS** | How spread out Y is *before* you fit anything — the total mess to be explained |
| **RSS** | How much mess is *still there* after fitting — the leftover |
| **R²** | The fraction of the mess you cleaned up |

R² runs 0 to 1. Near 1 = the model explains most variation. Near 0 = it explains almost nothing. **For the TV–sales model in the slides, R² = 0.61.**

> **Metaphor:** TSS is the size of the mess in your room before cleaning. RSS is what's still on the floor after you've tidied. R² is the percentage of the mess you actually dealt with. Note that this says nothing about *how big the room was* — a spotless 95% cleanup of a tiny room may still leave you with more absolute dirt than a 60% cleanup of a huge one. That absolute leftover is **RSE**. **Always read R² and RSE together.**

**Two facts to memorise:**
- In *simple* regression, R² = (correlation between X and Y)².
- **R² can never go down when you add a predictor** — even a column of random numbers will nudge it up. This is exactly why Adjusted R² exists.

## 3.2 Adjusted R²

Adjusted R² penalises you for each extra predictor. It **can** go down if a new variable doesn't earn its keep.

> **Metaphor:** R² is a mate who says "sure, bring anyone to the party." Adjusted R² is the one who asks "did that person actually contribute anything, or just eat the snacks?"

```python
print(model_tv.rsquared, model_tv.rsquared_adj)

# From the definition:
rss = model_tv.ssr                        # residual sum of squares
tss = model_tv.centered_tss               # total sum of squares
print(1 - rss / tss)                      # same as .rsquared

# In SIMPLE regression only, R2 == correlation squared:
print(df["TV"].corr(df["Sales"])**2)

# Proof that R2 always rises — add pure noise as a predictor:
df["junk"] = np.random.normal(size=len(df))
m_junk = sm.OLS(df["Sales"], sm.add_constant(df[["TV", "junk"]])).fit()
print(f"R2      {model_tv.rsquared:.4f} -> {m_junk.rsquared:.4f}   (rose)")
print(f"Adj-R2  {model_tv.rsquared_adj:.4f} -> {m_junk.rsquared_adj:.4f}   (fell)")
```

---

# PART 4 — Multiple linear regression

$$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \cdots + \beta_pX_p + \epsilon$$

For advertising: $\text{sales} = \beta_0 + \beta_1 \times \text{TV} + \beta_2 \times \text{radio} + \beta_3 \times \text{newspaper} + \epsilon$

**Critical interpretation:** $\beta_j$ is the average effect on Y of a one-unit increase in $X_j$, **holding all other predictors fixed**.

> **Metaphor:** That "holding all others fixed" clause is the whole ball game. It's the difference between *"people who drink more coffee sleep worse"* and *"if I gave this specific person one more coffee and changed nothing else about their life, their sleep would worsen by X."* Multiple regression tries to buy you the second statement. It only succeeds if the predictors can genuinely move independently — which is precisely what **collinearity destroys** (Part 6).

We estimate by minimising the same RSS, now over a plane/hyperplane:

$$\text{RSS} = \sum_{i=1}^{n}(y_i - \hat{\beta}_0 - \hat{\beta}_1x_{i1} - \cdots - \hat{\beta}_px_{ip})^2$$

```python
# Matrix API — pass a list of columns
X2 = sm.add_constant(df[["TV", "Radio"]])
model2 = sm.OLS(df["Sales"], X2).fit()
print(model2.summary().tables[1])

# Formula API — often cleaner, and handles categoricals automatically
model2 = smf.ols("Sales ~ TV + Radio", data=df).fit()

# Interaction / synergy ("Is there synergy among the media?" — slide 3)
# TV * Radio expands to: TV + Radio + TV:Radio
model_int = smf.ols("Sales ~ TV * Radio", data=df).fit()
print(model_int.pvalues["TV:Radio"])      # small p => the media reinforce each other

# Other useful formula syntax:
#   "Sales ~ TV + I(TV**2)"     polynomial term
#   "np.log(Sales) ~ TV"        transform the response
#   "Sales ~ TV + C(Region)"    treat Region as categorical
#   "Sales ~ . - News"          NOT supported in patsy — list columns explicitly
```

## 4.1 The four important questions

The slides organise multiple regression around these:

1. **Is at least one predictor useful?** → F-test
2. **Do all predictors help, or only a subset?** → variable selection
3. **How well does the model fit?** → R², RSE
4. **What should we predict, and how accurate is it?** → prediction + intervals

## 4.2 Question 1 — The F-statistic

$$H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0 \qquad H_A: \text{at least one } \beta_j \neq 0$$

$$F = \frac{(\text{TSS} - \text{RSS})/p}{\text{RSS}/(n - p - 1)} \sim F_{p,\, n-p-1}$$

Large F with a small p-value → at least one predictor is genuinely related to Y. If $H_A$ is true we expect F > 1. **The slides quote F = 570 for the advertising model** — overwhelming evidence.

> **Why can't I just look at the individual t-tests?** **Metaphor:** the F-test is the **building's smoke alarm**; the t-tests are **checking each room**. If you check 100 rooms individually at the 5% level, roughly 5 will look "suspicious" purely by chance — you'll find fake fires. The F-test asks one global question first, so it isn't fooled by multiple comparisons. **Check the alarm before you search rooms**, especially when p is large.

For very large p, or p > n, the F-test breaks down and we need forward selection instead.

```python
print(f"F = {model2.fvalue:.1f}   p = {model2.f_pvalue:.2e}")

# From the definition:
tss, rss, p, n = model2.centered_tss, model2.ssr, 2, len(df)
print(((tss - rss) / p) / (rss / (n - p - 1)))

# Comparing NESTED models — does the bigger model justify itself?
model3 = smf.ols("Sales ~ TV + Radio + News", data=df).fit()
f_stat, p_val, df_diff = model3.compare_f_test(model2)
print(f"F = {f_stat:.2f}, p = {p_val:.3f}")   # large p => keep the simpler model
```

## 4.3 Question 2 — Variable selection

- **Best subsets:** fit every possible subset of predictors, then choose using a criterion balancing fit against model size.
- **The problem:** there are $2^p$ subsets. At p = 40, that's over a **trillion** models (the slides say "over a billion" — either way, hopeless).
- **The fix:** automated searches — **forward selection** and **backward selection** (covered next week).

## 4.4 Question 3 — Model selection criteria *(flagged [Advanced])*

Later formal criteria for picking a model along a forward/backward path:
- Mallow's $C_p$
- Akaike Information Criterion (**AIC**)
- Bayesian Information Criterion (**BIC**) — punishes complexity hardest
- **Adjusted R²**
- **Cross-validation (CV)** — the most direct: just measure out-of-sample error

> All five are variations on one sentence: *"good fit, minus a fine for complexity."* They only disagree on the size of the fine.

## 4.5 Question 4 — Interpreting coefficients

**Ideal scenario — uncorrelated predictors (a balanced design):**
- Each coefficient can be estimated and tested separately.
- "A unit change in $X_j$ is associated with a $\beta_j$ change in Y, all else fixed" is a valid statement.

**When predictors are correlated:**
- Variance of *all* coefficients inflates, sometimes dramatically.
- Interpretation becomes hazardous — when $X_j$ changes, everything else changes too.
- **Claims of causality should be avoided for observational data.**

---

# PART 5 — Qualitative predictors

Some predictors are **categorical / factor variables**, not numeric: gender, student status, marital status, ethnicity, home ownership.

**Solution — dummy coding.** For house ownership:

$$x_i = \begin{cases} 1 & \text{if person } i \text{ owns a house} \\ 0 & \text{otherwise} \end{cases}$$

$$y_i = \beta_0 + \beta_1 x_i + \epsilon_i = \begin{cases} \beta_0 + \beta_1 + \epsilon_i & \text{owner} \\ \beta_0 + \epsilon_i & \text{non-owner} \end{cases}$$

So: **$\beta_0$ = the baseline group's mean**, and **$\beta_1$ = the difference between groups**.

> **Metaphor:** A dummy variable is a **light switch**. $\beta_0$ is the room's brightness with the switch off. $\beta_1$ is how much brighter it gets when you flip it on. It doesn't dim gradually — it's on or off.

### Credit card worked example (from the slides)

| Coefficient | Estimate | Std. Error | t | p |
|---|---|---|---|---|
| Intercept | 509.80 | 33.13 | 15.389 | < 0.0001 |
| own[Yes] | 19.73 | 46.05 | 0.429 | 0.6690 |

**Read it:** non-owners average \$509.80 balance; owners average \$509.80 + \$19.73 = \$529.53. But p = 0.669 → **the difference is not significant**. The \$19.73 gap is well within noise (SE of 46.05 dwarfs it).

**The coding is arbitrary and doesn't change the fit.** Flip the coding (non-owners = 1) and you get intercept 529.53 and slope −19.73 — the *same* two predictions, 509.80 and 529.53, just relabelled.

> **Metaphor:** Measuring temperature from the freezing point of water vs. from absolute zero. The numbers on the dial change; the weather doesn't.

**Rule to memorise:** a categorical variable with **K levels needs K−1 dummies**. The omitted level becomes the **baseline**, absorbed into the intercept. Include all K and you get perfect collinearity (the "dummy variable trap") — the model won't fit.

```python
# Easiest route — C() marks a column as categorical, dummies are automatic
m_own = smf.ols("Balance ~ C(Owner)", data=credit).fit()
print(m_own.summary().tables[1])
# Output row reads 'C(Owner)[T.Yes]' — the "T." means "treatment contrast
# vs the baseline". Baseline here is 'No', absorbed into the Intercept.

# A 3-level category automatically produces 2 dummies:
m_reg = smf.ols("Balance ~ C(Region)", data=credit).fit()

# Choose which level is the baseline:
smf.ols("Balance ~ C(Region, Treatment(reference='West'))", data=credit).fit()

# Manual route, if you want an explicit design matrix (e.g. for sklearn):
dummies = pd.get_dummies(credit["Region"], prefix="Region",
                         drop_first=True,   # <-- avoids the dummy variable trap
                         dtype=float)
print(dummies.columns)                      # 2 columns from 3 levels

# Mixing categorical and numeric, plus an interaction between them:
smf.ols("Balance ~ Income + C(Owner)", data=credit).fit()          # parallel lines
smf.ols("Balance ~ Income * C(Owner)", data=credit).fit()          # different slopes too
```

> **Reading the output label.** `C(Owner)[T.Yes]` trips people up. It means "the effect of being in group *Yes* **relative to the baseline**". It is a *difference*, never a group mean. To recover the group means: baseline = `Intercept`, other group = `Intercept + coefficient`.

---

# PART 6 — Potential problems (diagnostics)

The slides list three. These are the "**is the ruler even appropriate here?**" checks.

## 6.1 Non-linearity

**Tool:** plot residuals $e_i$ vs fitted values $\hat{y}_i$.

- **Random scatter around zero** → linear model is fine.
- **A U-shape or any curve** → the model is systematically missing something. Consider polynomial terms, or transform X.

> **Metaphor:** A residual plot is an **X-ray**. The scatterplot shows you the patient; the residual plot shows you the broken bone. Any *shape* in the residuals is signal that leaked out of your model instead of being captured by it. **Residuals should look boring.** Boring is the goal.

```python
def diagnostics(model, name):
    """Residuals-vs-fitted + Normal Q-Q, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].scatter(model.fittedvalues, model.resid, alpha=0.4, s=20)
    axes[0].axhline(0, color="red", ls="--")
    axes[0].set(xlabel="Fitted", ylabel="Residuals",
                title=f"{name} — Residuals vs Fitted")

    sm.qqplot(model.resid, line="s", ax=axes[1], alpha=0.5)
    axes[1].set_title("Normal Q-Q")

    plt.tight_layout()
    plt.show()

diagnostics(model_tv, "TV")

# The fix when you see a curve — add a quadratic term:
m_poly = smf.ols("Sales ~ TV + I(TV**2)", data=df).fit()
print(m_poly.pvalues["I(TV ** 2)"])   # significant => the curvature is real
```

## 6.2 Non-constant variance (heteroscedasticity)

The linear model assumes $\text{Var}(\epsilon_i) = \sigma^2$ — constant for every observation.

- **Fan / megaphone shape** in the residual plot → variance grows with fitted values.
- **Remedy:** transform Y — natural log, or a Box–Cox transformation.

> **Metaphor:** A weather forecaster who's accurate to ±1°C in mild conditions but ±15°C in storms. The forecast isn't *biased* — it's not systematically too hot or too cold — but the **error bars are lying**, because they're quoted as one fixed number. That's the real damage of heteroscedasticity: your coefficients stay roughly unbiased, but your **SEs, CIs and p-values become untrustworthy**.
>
> **Why does a log fix it?** Logging compresses the big end of the scale hardest, so it squashes the wide end of the megaphone back down to match the narrow end.

```python
# Formal test — small p means non-constant variance
lm, lm_p, f, f_p = het_breuschpagan(model.resid, model.model.exog)
print(f"Breusch-Pagan p = {lm_p:.4f}",
      "-> HETEROSCEDASTIC" if lm_p < 0.05 else "-> constant variance OK")

# Remedy 1: transform the response
m_log = smf.ols("np.log(Sales) ~ TV + Radio", data=df).fit()

# Remedy 2: keep the model, but use robust standard errors.
# Coefficients are unchanged; only the SEs, t-stats and p-values are corrected.
m_robust = smf.ols("Sales ~ TV + Radio", data=df).fit(cov_type="HC3")
print(m_robust.summary().tables[1])
```

> Remedy 2 isn't in the slides, but it's what most practitioners reach for first: heteroscedasticity doesn't bias your coefficients, it only corrupts your standard errors — so you can just fix the standard errors directly and leave the model alone.

## 6.3 Collinearity

Two or more predictors are closely related to each other.

**Effect:** inflates standard errors, making it hard to isolate each predictor's individual contribution. Coefficients become unstable and can flip sign with tiny data changes.

**Detection — Variance Inflation Factor:**

$$\text{VIF}(\hat{\beta}_j) = \frac{1}{1 - R^2_{X_j | X_{-j}}}$$

That $R^2$ is from regressing $X_j$ on **all the other predictors**. If the others predict $X_j$ well, $R^2 \to 1$ and VIF explodes.

| VIF | Verdict |
|---|---|
| ≈ 1 | No collinearity |
| > 5 | Moderate concern |
| > 10 | Serious problem |

**Remedies:** drop one of the collinear variables, or combine them (e.g. average them into a single index).

```python
def vif_table(frame, cols):
    X = frame[cols].assign(const=1)       # constant column is REQUIRED here,
    out = pd.DataFrame({                  # otherwise every VIF comes out wrong
        "Feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })
    return out[out.Feature != "const"].round(3)

print(vif_table(df, ["TV", "Radio", "News"]))

# Verify the formula by hand — regress one predictor on all the others:
aux = sm.OLS(df["News"], sm.add_constant(df[["TV", "Radio"]])).fit()
print(1 / (1 - aux.rsquared))             # matches the table

# Quick first look before any of this — just eyeball the correlations:
print(df[["TV", "Radio", "News"]].corr().round(3))
```

> ⚠️ **The `assign(const=1)` line is the trap.** `variance_inflation_factor` assumes your matrix already contains an intercept column. Omit it and the VIFs are silently, badly wrong — usually inflated to absurd values. Add it, compute, then filter the constant row back out of the display.

> **Metaphor:** Two people carry a couch up the stairs together, every single time, never separately. The couch arrives — the *prediction* is fine. But ask "how much did each person lift?" and you genuinely cannot know. Any split (70/30, 50/50, 110/−10) is consistent with what you observed. That uncertainty about the split *is* the inflated standard error.
>
> **The crucial nuance:** collinearity hurts **interpretation**, not **prediction**. If you only care about forecasting sales, collinearity is nearly harmless. If you care about *which channel to fund*, it's fatal.

---

# PART 7 — The lab, worked through

Simulated data, n = 200, seeded at 42. The **true data-generating process** was:

$$\text{Sales} = 2.9 + 0.046\,\text{TV} + 0.188\,\text{Radio} + \mathcal{N}(0, 1.5)$$

and crucially: `News = 0.3 * TV + uniform noise` — **News was deliberately built to be a shadow of TV.** Radio was generated independently.

> This is the lab's whole point: it plants a fake variable that *looks* informative and challenges you to catch it.

**Tools used:** `statsmodels` (`sm.OLS`, `sm.add_constant`, `variance_inflation_factor`, `qqplot`) and `sklearn` (`train_test_split`, `mean_squared_error`, `r2_score`).

⚠️ **Don't forget `sm.add_constant(X)`** — unlike sklearn, statsmodels does **not** add an intercept for you. Omit it and you force the line through the origin.

## Part 1 — Simple OLS

**Tutorial: Sales ~ TV**

| | coef | std err | t | P>\|t\| |
|---|---|---|---|---|
| const | 7.7109 | 0.414 | 18.611 | 0.000 |
| TV | 0.0449 | 0.002 | 18.415 | 0.000 |

**R² = 0.631, RSE = 3.04**

**Exercise 1: Sales ~ Radio**

| | coef | std err | t | P>\|t\| |
|---|---|---|---|---|
| const | 9.8958 | 0.611 | 16.205 | 0.000 |
| Radio | 0.1719 | 0.021 | 8.205 | 0.000 |

**R² = 0.254, RSE = 4.33**

**Answers:** each extra unit of Radio spend is associated with ~0.172 more units of Sales. **TV explains far more variance** (0.631 vs 0.254) and has a smaller typical miss (3.04 vs 4.33).

> **Don't be fooled by the slope sizes.** Radio's slope (0.172) is nearly *four times* TV's (0.045), yet TV is the better predictor. Slope magnitude depends on the units of X — Radio runs 0–50, TV runs 0–300. **A slope is a rate, not an importance score.**

### 🔍 A subtle gem hidden in these numbers

The true intercept was **2.9**, but the TV-only model estimates **7.71**. Why?

Dropping Radio doesn't destroy its effect — it gets absorbed. Mean Radio ≈ 25.2, and 0.188 × 25.2 ≈ 4.7. And 2.9 + 4.7 ≈ **7.6 ≈ 7.71** ✓

Meanwhile the TV slope stays honest (0.0449 vs true 0.046) — **because TV and Radio are uncorrelated**, omitting Radio doesn't bias TV's slope, it just dumps Radio's average contribution into the intercept. That's **omitted variable bias**, and it's the cleanest illustration of it you'll get.

## Part 2 — Residual diagnostics

Both models get **residuals vs fitted** + **Normal Q-Q** plots.

- Residuals vs fitted → checks linearity and constant variance.
- Q-Q plot → checks normality of errors (points on the diagonal = normal).

**Findings:** residuals appear random for both → OLS assumptions roughly met. *(Expected — the data was simulated with clean Gaussian noise.)*

**Answer to task 3:** a systematic curve would suggest the true relationship is non-linear and the model is mis-specified — add polynomial terms or transform.

> **Metaphor:** The Q-Q plot is a **height chart against a doorframe**. If your data's quantiles match the normal distribution's quantiles, every mark lines up along the diagonal. Bends at the ends mean **fat tails** — more extreme values than a normal distribution would produce.

## Part 3 — Multiple regression

**Tutorial: Sales ~ TV + Radio**

| | coef | std err | t | P>\|t\| |
|---|---|---|---|---|
| const | 3.0724 | 0.283 | 10.858 | 0.000 |
| TV | 0.0457 | 0.001 | 37.270 | 0.000 |
| Radio | 0.1793 | 0.007 | 24.217 | 0.000 |

**R² = 0.907, Adj-R² = 0.906, F = 964.2, p(F) = 1.79e-102**

🎯 Compare to truth (2.9, 0.046, 0.188): **the model nailed it.** And R² leapt from 0.631 → 0.907 by adding one honest variable.

**Exercise 3: Sales ~ TV + Radio + News**

| | coef | std err | t | P>\|t\| |
|---|---|---|---|---|
| const | 2.8157 | 0.357 | 7.892 | 0.000 |
| TV | 0.0416 | **0.004** | 11.225 | 0.000 |
| Radio | 0.1804 | 0.007 | 24.197 | 0.000 |
| News | 0.0140 | 0.012 | 1.179 | **0.240** |

**R² = 0.908, Adj-R² = 0.907, F = 644.5**

**Answers:**
1. R² rose by a trivial 0.001 (it *had* to rise — it always does).
2. **F dropped from 964.2 → 644.5.** Adding a dead-weight predictor *dilutes* the F-statistic, because it divides the explained variation by a larger p.
3. **News is not significant** (p = 0.240 ≫ 0.05); its CI [−0.009, 0.037] straddles zero.
4. **Prefer TV + Radio** — simpler, stronger F, no meaningful loss of fit.

### 🔍 The detail worth marks in an exam

Look at TV's standard error: **0.001 → 0.004, a fourfold inflation.** And TV's coefficient drifted from 0.0457 to 0.0416, away from the true 0.046.

That is collinearity doing damage **in real time**, before you've even computed a VIF. News stole part of TV's credit and blurred it. *This is the couch metaphor made numeric.*

## Part 4 — Multicollinearity (VIF)

**Tutorial VIFs:**

| Feature | VIF |
|---|---|
| TV | 9.150 |
| Radio | 1.017 |
| **News** | **9.185** |

TV and News are locked together (both ≈ 9.2, well over 5); Radio is perfectly clean at ≈ 1.0 — exactly as designed.

**Exercise 4 — after dropping News:**

| Feature | VIF |
|---|---|
| TV | 1.0007 |
| Radio | 1.0007 |

**R² without News: 0.907 (with News: 0.908)**

**Answers:**
1. Both TV and News exceed 5 — but **News is the one to drop**, because News was *derived from* TV. TV is the genuine driver.
2. Yes, VIFs collapse to ≈ 1.
3. R² barely moves → **News carried no independent information.** It was a mirror, not a new window.

> **How do you choose which of the collinear pair to drop?** VIF alone can't tell you — it's symmetric. You use domain knowledge, theory, or which variable is cheaper/more reliable to measure. Here the answer is known because we built the data.

## Part 5 — Train/test evaluation

80/20 split, `random_state=0`.

| Model | Test RMSE |
|---|---|
| TV + Radio | 1.541 |
| TV + Radio + News | 1.526 |

**Answers:** the test RMSEs are **effectively identical** — the difference is noise on a 40-observation test set, not a real improvement. News does not earn its place. In-sample R² always favours the bigger model; out-of-sample error is the honest referee, and it says the extra predictor bought nothing.

🎯 **Also note:** test RMSE ≈ 1.53, and the true noise σ was **1.5**. The model has essentially reached the **irreducible error floor** — you cannot do better without new information. This is the ceiling every model runs into eventually.

> **Metaphor:** In-sample R² is **marking your own exam with the answer sheet open in front of you**. Test RMSE is **sitting the exam in a supervised hall**. The gap between the two is your overfitting.

**Metrics recap from the lab:**

| Metric | Meaning | Units | Direction |
|---|---|---|---|
| **MSE** | Mean of (prediction − actual)² | y² | Lower better |
| **RMSE** | √MSE — typical miss | Same as y | Lower better |
| **R²** | Fraction of variance explained | Unitless, 0–1 | Higher better |

RMSE is usually the one to report to a human — it's in the units they actually care about.

```python
y = df["Sales"]

def evaluate(cols, label, seed=0):
    X = sm.add_constant(df[cols])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    m = sm.OLS(y_tr, X_tr).fit()          # fit on TRAIN only
    pred = m.predict(X_te)                # predict on unseen TEST
    mse = mean_squared_error(y_te, pred)
    print(f"{label:20s} MSE {mse:6.3f} | RMSE {np.sqrt(mse):.3f} "
          f"| Test R2 {r2_score(y_te, pred):.3f}")
    return np.sqrt(mse)

evaluate(["TV", "Radio"], "TV + Radio")
evaluate(["TV", "Radio", "News"], "TV + Radio + News")
```

**One split is a coin flip.** Average over many to see the truth:

```python
import contextlib, io

with contextlib.redirect_stdout(io.StringIO()):        # silence the 60 print lines
    scores = [(evaluate(["TV", "Radio"], "", s),
               evaluate(["TV", "Radio", "News"], "", s))
              for s in range(30)]

print(pd.DataFrame(scores, columns=["TV+Radio", "+News"]).mean().round(3))
```

Averaged over 30 splits: **1.531** vs **1.538** — the simpler model wins. On the single split the lab uses, the News model happens to score marginally *better* (1.526 vs 1.541), which is exactly why you shouldn't trust one split. This variability is the argument for **cross-validation**, coming in Week 3.

---

# PART 8 — Where this goes next

**Generalizations of the linear model (slide 26):**

| Limitation | Extension | When you'll meet it |
|---|---|---|
| Classification instead of regression | Logistic regression, SVMs | Week 4 |
| Non-linearity | Kernel smoothing, splines, GAMs, kNN | Later |
| Interactions & non-linearity together | Trees, bagging, random forests, boosting | Later |
| Too many predictors / overfitting | Ridge regression, Lasso | **Week 3** |

Every one of these starts from the same skeleton: **predict, measure error, minimise it, check the residuals.**

---

# PART 9 — Making predictions with uncertainty

This answers advertising question 4: *given predictor values, what should we predict, and how accurate is it?*

```python
new = sm.add_constant(
    pd.DataFrame({"TV": [150.0, 250.0], "Radio": [25.0, 40.0]}),
    has_constant="add",                   # forces the const column even on small frames
)

pred = model2.get_prediction(new).summary_frame(alpha=0.05)
print(pred[["mean", "mean_ci_lower", "mean_ci_upper",
            "obs_ci_lower", "obs_ci_upper"]].round(2))

# Just the point predictions:
print(model2.predict(new))
```

Output:

| | mean | mean_ci_lower | mean_ci_upper | obs_ci_lower | obs_ci_upper |
|---|---|---|---|---|---|
| 0 | 14.41 | 14.20 | 14.62 | 11.39 | 17.44 |
| 1 | 21.67 | 21.27 | 22.07 | 18.63 | 24.71 |

**Two different intervals — a very common exam question:**

| Interval | Answers | Width |
|---|---|---|
| `mean_ci` — **confidence** | What is the *average* sales across all campaigns with this budget? | Narrow (±0.2) |
| `obs_ci` — **prediction** | What will *this one specific* campaign do? | Wide (±3.0) |

> **Metaphor:** Predicting the average height of Australian men is easy — average a big enough sample and you'll nail it. Predicting the height of *the next man to walk through the door* is a completely different problem, no matter how much data you have. The confidence interval shrinks as you collect data; **the prediction interval can never shrink below the noise floor** (σ = 1.5 here).

---

# Key takeaways (as the slides state them)

1. **OLS minimises RSS** and has closed-form estimates; use **t-tests** for individual coefficients and the **F-test** for joint significance.
2. **R²** measures proportion of variance explained; **RSE** estimates the standard deviation of the residuals.
3. **Qualitative predictors → dummy variables** (K levels ⇒ K−1 dummies); interaction terms model joint effects.
4. **Diagnose residuals** for non-linearity and heteroscedasticity; check **VIF > 5–10** for collinearity.
5. The same framework underpins **regularised regression (Week 3)** and **logistic regression (Week 4)**.

---

# Formula sheet

| Quantity | Formula |
|---|---|
| Simple model | $Y = \beta_0 + \beta_1 X + \epsilon$ |
| Prediction | $\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x$ |
| Residual | $e_i = y_i - \hat{y}_i$ |
| RSS | $\sum e_i^2$ |
| TSS | $\sum (y_i - \bar{y})^2$ |
| RSE | $\sqrt{\text{RSS}/(n-2)}$ |
| R² | $1 - \text{RSS}/\text{TSS}$ |
| t-statistic | $\hat{\beta}_1 / \text{SE}(\hat{\beta}_1)$ |
| 95% CI | $\hat{\beta}_1 \pm 2\,\text{SE}(\hat{\beta}_1)$ |
| F-statistic | $\dfrac{(\text{TSS}-\text{RSS})/p}{\text{RSS}/(n-p-1)}$ |
| VIF | $1/(1 - R^2_{X_j\mid X_{-j}})$ |
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ |
| Number of subsets | $2^p$ |

---

# statsmodels cheat sheet

Given `model = sm.OLS(y, X).fit()`:

| Formula / concept | Code |
|---|---|
| $\hat{\beta}_0, \hat{\beta}_1, \ldots$ | `model.params` |
| SE($\hat{\beta}_j$) | `model.bse` |
| t-statistics | `model.tvalues` |
| p-values | `model.pvalues` |
| 95% CI | `model.conf_int()` |
| RSS | `model.ssr` |
| TSS | `model.centered_tss` |
| R² | `model.rsquared` |
| Adjusted R² | `model.rsquared_adj` |
| **RSE** | `np.sqrt(model.mse_resid)` — **no built-in attribute** |
| F-statistic / its p-value | `model.fvalue` / `model.f_pvalue` |
| Residuals $e_i$ | `model.resid` |
| Fitted values $\hat{y}_i$ | `model.fittedvalues` |
| AIC / BIC | `model.aic` / `model.bic` |
| Degrees of freedom $n-p-1$ | `model.df_resid` |
| Coefficient table only | `model.summary().tables[1]` |
| Predict on new data | `model.predict(X_new)` |
| Predict with intervals | `model.get_prediction(X_new).summary_frame()` |
| Compare nested models | `big.compare_f_test(small)` |
| VIF | `variance_inflation_factor(X.values, i)` |
| Heteroscedasticity test | `het_breuschpagan(model.resid, model.model.exog)` |
| Robust standard errors | `.fit(cov_type="HC3")` |

### Three mistakes that cost marks

1. **Forgetting `sm.add_constant(X)`.** No error is raised — you just silently fit a line through the origin.
2. **Forgetting the constant column in `variance_inflation_factor`.** Also silent, also wrong.
3. **Argument order.** It's `sm.OLS(y, X)` — response first. sklearn is the opposite (`fit(X, y)`), which is a very easy thing to mix up when switching between them.

---

# Quick self-test

Cover the answers and try these.

1. **A model has R² = 0.95 but RSE = 4000 units. Is it good?** — *Depends entirely on the scale of Y. It explains 95% of variation, but if typical sales are 5000 units, you're still off by 80% on a typical prediction. Always read both.*
2. **You add a variable and R² rises by 0.002. Should you keep it?** — *R² rising proves nothing; it always rises. Check its p-value, Adjusted R², and out-of-sample error.*
3. **Two predictors each have p > 0.5, but the F-test is highly significant. What's happening?** — *Classic collinearity. Together they clearly explain Y, but each is individually redundant given the other, so neither can claim credit.*
4. **The residual plot fans out to the right. What now?** — *Heteroscedasticity. Coefficients are still roughly OK, but your SEs and p-values are unreliable. Try log(Y) or Box–Cox.*
5. **Why does "holding all else fixed" fail with correlated predictors?** — *Because in the data, the other predictors never actually stayed fixed while $X_j$ moved. You're extrapolating to a scenario you never observed.*
6. **You have a 4-level category. How many dummies?** — *Three. The fourth is the baseline, folded into the intercept. Use four and you hit the dummy variable trap.*
7. **Why can't you just check t-tests when p = 100?** — *Multiple comparisons — at α = 0.05 you'd expect ~5 false positives by chance. Run the F-test first.*
