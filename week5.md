# DSB102 — Week 5: Tree-Based Methods
### Complete study guide (lecture slides + solved lab)

---

## The one-paragraph version

Weeks 2–4 fitted **equations** — a global formula that applies everywhere in the feature space. Week 5 throws the equation away and fits a **partition** instead: chop the predictor space into rectangular boxes, and predict a constant inside each box (the mean for regression, the majority class for classification). That's a decision tree, and it's built by **recursive binary splitting** — greedily choosing the one split that most reduces error, then repeating inside each new region. Trees are wonderfully interpretable and **terrible on their own**, because a fully grown tree memorises the training set. Everything else this week is a fix for that. **Pruning** cuts a big tree back using a complexity penalty $\alpha$, chosen by cross-validation. **Bagging** averages hundreds of trees grown on bootstrap resamples, cancelling variance. **Random forests** improve bagging by forcing each split to consider only $m \approx \sqrt{p}$ random predictors, which stops every tree from looking the same. **Boosting** grows trees sequentially, each one fitted to what the previous ones got wrong.

> **Master metaphor — the diagnostic flowchart vs. the panel of doctors.** A single tree is a laminated flowchart on a clinic wall: *Years < 4.5? Go left. Hits < 117.5? Go right.* Anyone can follow it, which is exactly why trees are loved. But a flowchart written by one junior doctor from one afternoon of patients will be idiosyncratic — change the patients slightly and you get a completely different chart. **Bagging** is asking 500 doctors, each trained on a slightly different sample, and taking the vote. **Random forests** notice that all 500 doctors keep opening with the same question, so they force each one to consider a random subset of tests at every step — deliberately handicapping them so they *disagree*, because votes only help when the voters are independent. **Boosting** is different in kind: one doctor sees the patient, then a second is briefed only on what the first got wrong, then a third on what the first two still got wrong. Sequential specialists, not a committee.

**The thread from Weeks 2–4.** The bias–variance trade-off is the whole story again, but the dial has a new name each week: $p$ (Week 2), $\lambda$ (Week 3), $K$ (Week 4), and now **tree depth / $\alpha$**. The U-shaped test-error curve you've drawn four times now shows up again in Exercise 1. And cross-validation is *still* how you choose the dial.

---

## Setup — imports used throughout

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import (DecisionTreeRegressor, DecisionTreeClassifier, plot_tree)
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.metrics import mean_squared_error, accuracy_score
```

---

# PART 1 — What a tree actually is

Tree-based methods **segment the predictor space into regions**. The set of splitting rules used to do the segmenting can be drawn as a tree, which is where the name comes from. They work for both regression and classification, and the enhanced versions — bagging, random forests, boosting — are all built from them.

## 1.1 The baseball example (know this one cold)

Predicting **log salary** from `Years` and `Hits`. The final tree has **two internal nodes** and **three terminal nodes (leaves)**:

```
                Years < 4.5
               /            \
            5.11         Hits < 117.5
                        /            \
                     6.00           6.74
```

This partitions the predictor space into three regions:

$$R_1 = \{X \mid \text{Years} < 4.5\}, \quad R_2 = \{X \mid \text{Years} \ge 4.5,\ \text{Hits} < 117.5\}, \quad R_3 = \{X \mid \text{Years} \ge 4.5,\ \text{Hits} \ge 117.5\}$$

**The number in each leaf is the mean response of the training observations that land there.** So every player with fewer than 4.5 years of experience gets the identical prediction of 5.11, regardless of anything else about them.

**How the slides read this tree:**
- **Years is the most important factor** in determining salary — it's the *first* split, made when the whole training set was still available.
- For **inexperienced** players, Hits barely matters (that branch never splits again).
- For players with **5+ years**, Hits does matter, and more hits means higher salary.

> **Read the structure, not just the leaves.** Position in the tree *is* the interpretation. The root split is the single most useful question you can ask about a player; splits further down are refinements that only apply to the sub-population that reached them. This is genuinely different from a regression coefficient, which claims to apply to everyone at once. **A tree lets the effect of `Hits` exist for veterans and not exist for rookies** — an interaction it discovered without being told to look for one.

The slides concede this is "surely an over-simplification", but note that compared to a regression model it's **easy to display, interpret and explain**. That trade — accuracy for explicability — is the entire argument for single trees.

## 1.2 Vocabulary

| Term | Meaning |
|---|---|
| **Internal node** | A point where the space is split (a question) |
| **Branch** | The line connecting nodes |
| **Terminal node / leaf** | A final region $R_j$ where a prediction is made |
| **Depth** | Longest path from root to leaf |
| **Region $R_j$** | The box in predictor space corresponding to a leaf |

⚠️ Trees are drawn **upside down** — the root is at the top, leaves at the bottom.

---

# PART 2 — Growing a regression tree

## 2.1 The objective

1. Divide the predictor space (all possible values of $X_1, \dots, X_p$) into $J$ **distinct and non-overlapping** regions $R_1, \dots, R_J$.
2. For every observation falling in $R_j$, predict the **mean of the training responses in $R_j$**.
3. In theory the regions could be any shape; we use **high-dimensional rectangles ("boxes")** for simplicity and interpretability.
4. Choose the boxes to minimise

$$\sum_{j=1}^{J} \sum_{i \in R_j} \left(y_i - \hat{y}_{R_j}\right)^2$$

This is just **RSS from Week 2**, with the fitted value $\hat{y}_i$ replaced by "the mean of my box".

> **Why boxes?** Two reasons, and the second one matters more. The stated reason is interpretability — a box is a conjunction of simple rules (`Years ≥ 4.5 AND Hits < 117.5`), which reads as a sentence. The unstated reason is **tractability**: restricting to axis-aligned splits makes the search finite. Allow diagonal or curved boundaries and there is no algorithm; allow boxes and you can enumerate every candidate.

## 2.2 Recursive binary splitting

Trying every possible partition of the feature space into boxes is **computationally infeasible**. So we build the tree **top-down, one split at a time**:

- **Top-down** — start at the top with all the data, and each split creates two new branches further down.
- **Greedy** — at each step pick whichever split is best **right now**, rather than looking ahead to what might be best overall.

**The splitting search, in full:**

1. Try **every predictor** $j$ and **every possible cutpoint** $s$ for that predictor.
2. For each candidate, compute how much the split reduces RSS — i.e. how much better two groups fit than one did.
3. Take the $(j, s)$ pair with the **biggest improvement**.
4. Repeat **within each new region**, splitting again and again, until a stopping rule fires (minimum node size, maximum depth, or negligible improvement).

There are $p \times (n-1)$ candidate splits at each step, which is why this is fast.

> **Metaphor — playing Twenty Questions badly, on purpose.** A perfect player plans several questions ahead. The greedy algorithm asks whichever single question splits the remaining possibilities best *this turn*, with no plan. It's demonstrably suboptimal: a mediocre first split can occasionally unlock two brilliant second splits, and greedy will never find that. **We accept a worse tree in exchange for a tree we can actually compute.** This is the same bargain as forward stepwise selection in Week 3 — greedy, sequential, no backtracking, defensible only because the exhaustive alternative is impossible.

## 2.3 Worked example — finding the best split (slide 29)

Six observations, one predictor:

| $i$ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| $X$ | 1 | 2 | 3 | 4 | 5 | 6 |
| $Y$ | 2 | 3 | 9 | 10 | 11 | 12 |

**Candidate split at $s = 2.5$** (i.e. $X < 2.5$ vs $X \ge 2.5$):

**Left region** $R_1$ = obs {1, 2}, $\bar{y}_{R_1} = 2.5$:
$$RSS_1 = (2 - 2.5)^2 + (3 - 2.5)^2 = 0.25 + 0.25 = 0.5$$

**Right region** $R_2$ = obs {3, 4, 5, 6}, $\bar{y}_{R_2} = 10.5$:
$$RSS_2 = (9-10.5)^2 + (10-10.5)^2 + (11-10.5)^2 + (12-10.5)^2 = 2.25 + 0.25 + 0.25 + 2.25 = 5.0$$

**Total RSS = 0.5 + 5.0 = 5.5.**

In general we search all $p \times (n-1)$ splits and take the smallest total RSS.

> **I ran every candidate cutpoint, because the slides only show one and the exam may ask you to justify it:**
>
> | $s$ | 1.5 | **2.5** | 3.5 | 4.5 | 5.5 |
> |---|---|---|---|---|---|
> | Total RSS | 50.0 | **5.5** | 30.7 | 50.5 | 70.0 |
>
> $s = 2.5$ genuinely wins, and by a mile. The unsplit RSS (TSS) is 90.8, so **one split removes 94% of the variation**. The reason is visible in the data: there's a cliff between $Y=3$ and $Y=9$, and 2.5 is the only cutpoint that puts the cliff on the boundary rather than inside a region. **Splits chase discontinuities** — that's what they're for, and it's why trees beat linear models on step-shaped relationships.

## 2.4 Full regression tree algorithm (slide 14 — memorise the four steps)

1. Use **recursive binary splitting** to grow a large tree, stopping only when each terminal node has fewer than some minimum number of observations.
2. Apply **cost complexity pruning** to get a sequence of best subtrees as a function of $\alpha$.
3. Use **K-fold cross-validation** to choose $\alpha$. For each fold $k$: repeat steps 1–2 on the other $\frac{K-1}{K}$ of the data, evaluate MSE on the held-out fold as a function of $\alpha$. Average across folds and pick the $\alpha$ minimising the average error.
4. Return the subtree from step 2 corresponding to that $\alpha$.

> ⚠️ **Note the order: grow big, then cut back.** Why not just stop splitting early when improvement gets small? Because a split that looks worthless can enable a very good split beneath it — the greedy algorithm is short-sighted, so early stopping compounds its blindness. **Growing first and pruning second lets you see the whole tree before deciding what to cut.** This is a favourite short-answer question.

---

# PART 3 — Pruning

Large trees **overfit**. Pruning cuts a big tree back to a smaller subtree.

Checking every possible subtree isn't feasible, so we use **cost-complexity pruning** (a.k.a. **weakest-link pruning**), scoring each candidate subtree $T$ by

$$C_\alpha(T) = \underbrace{\sum_{m=1}^{|T|} \sum_{i:\, x_i \in R_m} (y_i - \hat{y}_{R_m})^2}_{\text{how well } T \text{ fits}} + \underbrace{\alpha |T|}_{\text{how big } T \text{ is}}$$

where $|T|$ is the number of terminal nodes.

**$\alpha$ is a knob for tree size:**

| $\alpha$ | Effect |
|---|---|
| $\alpha = 0$ | No penalty — keep the **full tree** |
| $\alpha$ small | Light pruning |
| $\alpha$ large | Every extra leaf "costs" more, so more branches get pruned |
| $\alpha \to \infty$ | Prune to a single root node (predict the overall mean) |

$\alpha$ controls the trade-off between complexity and training fit, and we pick it by **cross-validation**.

> **This is Ridge and Lasso wearing a different hat.** Compare the two objectives:
>
> $$\text{Lasso: } RSS + \lambda \sum |\beta_j| \qquad \text{Tree: } RSS + \alpha |T|$$
>
> Identical architecture — fit term plus complexity penalty, with a tuning parameter chosen by CV. The only change is **what counts as complexity**: for Lasso it's the size of the coefficients, for a tree it's the number of leaves. Once you see this, three weeks of material collapse into one idea: **regularisation is adding a price tag to complexity so the data has to justify buying it.**

> **Metaphor — a hedge and a pair of shears.** You let the hedge grow wild all summer, then trim. $\alpha$ is how ruthless you're feeling. Trim nothing and you keep every straggling twig (every idiosyncrasy of the training data). Trim hard and you're left with a stump that has no shape at all. Cross-validation is standing back and squinting to see which trim actually looks best from a distance.

## The baseball pruning example (slides 15–17)

- Data split in half: **132 training**, 131 test observations.
- Grew a large tree, varied $\alpha$ to produce subtrees with different numbers of terminal nodes.
- Ran **six-fold cross-validation** to estimate CV MSE as a function of $\alpha$.

**Result: the minimum cross-validation error occurs at a tree size of three.**

Read the figure carefully — it's the Week 3 diagnosis in a new setting:

| Curve | Behaviour |
|---|---|
| **Training MSE** | Falls monotonically as the tree grows. Never turns around. |
| **CV MSE** | Falls to a minimum at 3 leaves, then **rises**. |
| **Test MSE** | Tracks the CV curve closely — which is the whole point of CV. |

> **Training error can never select a model.** It always prefers the biggest tree, exactly as $R^2$ always prefers more predictors in Week 2 and training RSS always prefers the largest subset in Week 3. **The criterion that turns around is the only criterion that's telling you anything.** And note how well CV tracks test error here — that agreement is the empirical justification for the entire method.

> **Also note how small the winner is.** Out of a tree that could have had dozens of leaves, CV chooses **three**. Trees overfit far more aggressively than most students expect, and the honest pruned tree is usually smaller than it feels like it should be.

---

# PART 4 — Classification trees

Everything above transfers, with one change: **the prediction for a region is the most common class** among the training observations in it, and RSS no longer makes sense as a splitting criterion.

## 4.1 Why not just use the error rate?

The obvious criterion is the **classification error rate** — how often the tree's prediction would be wrong in that region. It's intuitive, but the slides call it **too crude**: it doesn't reward splits that make a region purer unless the split actually changes the **majority class**.

> **Concretely, here's the failure.** A node with 80 A's and 20 B's has an error rate of 0.20. Split it into (40 A, 0 B) and (40 A, 20 B). The first child is now *perfectly pure* — a genuine achievement. But the error rate is $0 \times 0.4 + 0.33 \times 0.6 = 0.20$, **exactly unchanged**. The criterion is blind to real progress because A is still the majority everywhere. Gini and cross-entropy both register the improvement, which is why they're used for *growing*.
>
> ⚠️ **But the error rate is preferable for pruning**, when the goal is predictive accuracy of the final tree rather than finding good splits. Growing and pruning can use different criteria — a subtle point worth a mark.

## 4.2 Gini index and cross-entropy

Both measure **node purity** — how mixed the classes are in a region. Let $\hat{p}_{mk}$ be the proportion of class $k$ in node $m$.

**Gini index:**
$$G = \sum_{k=1}^{K} \hat{p}_{mk}(1 - \hat{p}_{mk})$$

**Cross-entropy:**
$$D = -\sum_{k=1}^{K} \hat{p}_{mk} \log \hat{p}_{mk}$$

Both come out **near 0 when the node is pure** and are **largest when classes are evenly split**. At each split we choose whichever option produces the purest child nodes. The two give very similar trees in practice; **Gini is the more common default** (and sklearn's).

> **Metaphor — reaching into a bag of marbles.** The Gini index is the probability that if you draw one marble at random and then guess its colour by drawing a *second* marble at random, you guess wrong. Pure bag → you can't be wrong → $G = 0$. Fifty-fifty bag → you're wrong half the time → $G = 0.5$. **Gini is literally an expected error rate under random guessing**, which is why it's smooth: it responds to every shift in the proportions, not just to changes in which colour dominates.

## 4.3 Gini worked example (slide 30)

Node with 10 observations: **7 Class A, 3 Class B**. So $\hat{p}_A = 0.7$, $\hat{p}_B = 0.3$.

$$G = 0.7 \times 0.3 + 0.3 \times 0.7 = 0.21 + 0.21 = \mathbf{0.42}$$

A **pure node gives $G = 0$**; **maximum impurity gives $G = 0.5$** for $K = 2$.

**After splitting:**
- Left child (5 obs): 5 A, 0 B $\Rightarrow$ $G_L = 0$ **(pure)**
- Right child (5 obs): 2 A, 3 B $\Rightarrow$ $G_R = 2 \times 0.4 \times 0.6 = 0.48$

**Weighted Gini after the split:**
$$\frac{5}{10} \times 0 + \frac{5}{10} \times 0.48 = \mathbf{0.24} \;<\; 0.42$$

The split improves purity.

> ⚠️ **Weight by node size — this is the step people drop.** The naive average would be $(0 + 0.48)/2 = 0.24$, which happens to agree here *only because the children are the same size*. Split 9-and-1 and the unweighted average is badly wrong. Always use $\frac{n_L}{n}G_L + \frac{n_R}{n}G_R$.
>
> **The quantity being maximised is the *reduction*:** $0.42 - 0.24 = 0.18$. That's exactly what sklearn's feature importances accumulate (Part 7).

## 4.4 The Heart data example (slides 21–22)

- Binary outcome **HD (heart disease)** for **303 patients** presenting with chest pain. `Yes` = disease confirmed by angiographic test.
- **13 predictors** including Age, Sex, Chol, and other heart and lung function measures.
- **Cross-validation yields a tree with six terminal nodes.**

The figure shows the same three-curve story as baseball: training error keeps falling, CV and test error flatten and then drift up. The pruned six-node tree is a drastic simplification of the unpruned one — and performs as well.

> **Note that the tree splits on categorical predictors directly** (`Thal:a`, `ChestPain:bc`) with no dummy coding required. Trees handle qualitative predictors natively by splitting the *levels* into two groups — one of their genuine practical advantages over the regression machinery of Weeks 2–4.

---

# PART 5 — Linear models vs trees

Neither dominates. It depends entirely on the shape of the truth (slide 18):

| True boundary | Winner | Why |
|---|---|---|
| **Linear** | Linear model | A tree must approximate a diagonal with a staircase of axis-aligned steps — it needs many splits and still gets it wrong near the boundary |
| **Non-linear / blocky** | Tree | The tree reproduces a rectangular boundary exactly, with two splits; a linear model can't bend |

> **Metaphor — cutting a cake.** A linear model is one straight cut of any angle, all the way across. A tree gets unlimited cuts, but every cut must be parallel to an edge and only within the piece you're currently working on. If the pattern you want is a diagonal, the single angled cut wins easily and the tree produces a jagged staircase. If the pattern is a set of rectangular blocks, the tree nails it and the straight cut can't get close.

**How to actually decide:** cross-validation. This is not a philosophical question; estimate the test error of both and pick the winner.

**The other axis is interpretability.** A small tree is easier to explain to a non-technical audience than a regression equation — you can hand someone a flowchart. But that advantage evaporates the moment you move to a random forest, which is where the accuracy is. **You usually end up choosing between an interpretable tree and an accurate forest, not between trees and linear models.**

---

# PART 6 — Bagging

**Bootstrap aggregation** is a variance-reduction technique.

**The logic, in three steps:**
1. Averaging a set of models reduces variance. (Averaging $B$ independent quantities each with variance $\sigma^2$ gives variance $\sigma^2/B$.)
2. But in practice we have **one** training set, not many.
3. **Bootstrapping fixes this:** create many training sets by **resampling with replacement** from the one we have, fit a tree to each, and average the predictions. For classification, take a **majority vote**.

**Why trees specifically?** Each deep tree has **high variance but low bias**. Averaging cancels the variance while keeping the low bias — trees are the ideal raw material for this, because they're exactly the model whose weakness averaging repairs.

> ⚠️ **Bagged trees are grown deep and are NOT pruned.** This looks wrong after Part 3, but it's deliberate: you *want* each tree to be a low-bias, high-variance mess, because the averaging is what handles the variance. Pruning each tree first would add bias that averaging cannot remove. **Prune a single tree; never prune the members of an ensemble.**

> **Metaphor — the jellybean jar.** Ask one person how many jellybeans are in the jar and the answer is wildly off. Ask five hundred people and average: the errors are in every direction and largely cancel, leaving an estimate that is often startlingly accurate. Each guesser is unbiased-but-noisy, which is precisely the profile of a deep unpruned tree.

**What bagging costs you:** interpretability. There is no longer a flowchart to look at. This is why variable importance measures (Part 7) exist — they're the compensation.

---

# PART 7 — Random forests

Random forests improve on bagged trees by **decorrelating** them, which reduces variance further.

**The algorithm:**
1. Build decision trees on **bootstrapped samples** (as in bagging).
2. At **each split**, randomly select **$m$ predictors out of $p$**.
3. Choose the best split from **only those $m$**.
4. Typically $m \approx \sqrt{p}$ — e.g. **4 out of 13** for the Heart data.

**If $m = p$, a random forest is just bagging.** Bagging is the special case where you never restrict the candidate predictors.

> **Why on earth would you hide predictors from the algorithm?** This is the single most counter-intuitive idea in the topic, and it has a precise answer.
>
> The variance of an average of $B$ quantities with **pairwise correlation $\rho$** is
> $$\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$
> The second term vanishes as $B \to \infty$. **The first term does not.** Correlation between the trees puts a hard floor under how much averaging can help — and no number of extra trees will lower it.
>
> Now the problem with bagging: suppose one predictor is very strong. **Every bootstrap sample contains it, so nearly every tree splits on it first**, and the trees end up looking like near-copies of each other. Their errors are correlated, so averaging cancels much less than you'd hope.
>
> Forcing each split to ignore a random $p - m$ of the predictors means the strong one is unavailable roughly $(p-m)/p$ of the time, so other predictors get to lead. **Each individual tree is worse; the forest is better.** You are trading a little bias for a large drop in $\rho$.

> **Metaphor — the committee where everyone reads the same newspaper.** Five hundred forecasters who all read the same front page will make the same mistake five hundred times, and averaging them is barely better than asking one. Ban each forecaster from reading a random subset of the sources and they start noticing different things. Their individual forecasts get slightly worse and the consensus gets substantially better. **Diversity in the voters, not skill in each voter, is what makes voting work.**

## 7.1 Variable importance

For bagged/RF regression (or classification) trees, we record the **total amount that the RSS (or Gini index) is decreased due to splits over a given predictor, averaged over all $B$ trees**. A large value indicates an important predictor (usually reported normalised to 100).

On the Heart data, the ranking runs **Thal** (100) > **Ca** > **ChestPain** > **Oldpeak** > **MaxHR**, down to **Fbs** at essentially zero.

> ⚠️ **Importance is not a coefficient.** It has no sign — it tells you a predictor is *used*, never whether it pushes the prediction up or down. It's also known to be **biased toward high-cardinality and continuous predictors**, which simply offer more candidate cutpoints to win with; a binary predictor is at a structural disadvantage. And with two highly correlated predictors, the importance gets **split between them**, so both can look mediocre when the underlying signal is strong. Use it to generate hypotheses, not to conclude.

---

# PART 8 — Boosting

| | Bagging / Random forests | Boosting |
|---|---|---|
| How trees are built | **Independently**, in parallel | **Sequentially** |
| What each tree sees | A bootstrap resample of the data | The **mistakes (residuals)** of the trees before it |
| Combination | Average / majority vote | Weighted sum, scaled by a learning rate |
| Tree size | Deep, unpruned | **Small** — often stumps or depth 2–3 |
| Main risk | Very little; more trees never hurts | **More trees eventually overfits** |

Boosting **learns slowly and often achieves very high accuracy**. The slides don't go through the algorithm in detail, but the shape is: fit a small tree, look at what's still wrong, fit the next small tree to *that*, add a shrunken version of it to the running prediction, repeat.

**Popular fast, regularised implementations: XGBoost and LightGBM.** These dominate tabular-data competitions.

> **Metaphor — sculpting vs. surveying.** Bagging is a survey: everyone answers the same question independently and you average. Boosting is sculpting: each pass removes a little more of what's left of the error, and **the order matters absolutely** — you cannot run pass 40 before pass 39, because pass 40 is defined in terms of what pass 39 left behind. This is why bagging parallelises trivially and boosting doesn't, and why bagging is robust to over-tuning while boosting will happily carve straight through your block of marble if you let it.

**The three knobs (all interacting):**

| Parameter | sklearn | Effect |
|---|---|---|
| Number of trees $B$ | `n_estimators` | **Too many overfits.** Choose by CV or early stopping |
| Learning rate $\lambda$ | `learning_rate` | How much of each tree gets added. Small = slow, careful learning |
| Tree depth $d$ | `max_depth` | Interaction order. $d=1$ (stumps) = additive model, no interactions |

> **$B$ and the learning rate trade off directly.** Halve the learning rate and you need roughly twice the trees. **Small learning rate + many trees is almost always better than the reverse** — that's what "learns slowly" means. You'll see this in the lab, where the default settings overfit badly and a ten-times-smaller learning rate produces the best model of the entire week.

## The spam data comparison (slide 28)

Test error against number of trees, for the same dataset:

| Method | Converged test error |
|---|---|
| Bagging | ≈ 0.054 |
| Random Forest | ≈ 0.049 |
| **Gradient Boosting (5-node)** | **≈ 0.045** |

The ordering **bagging > RF > boosting** (in error) matches the story: decorrelation beats plain averaging, and sequential correction beats both — *when boosting is tuned properly*. Note also that bagging and RF **flatten out and stay flat** — more trees never hurt them. Boosting's curve is the one you have to watch.

---

# PART 9 — The lab, worked through

**Two datasets, both from sklearn:**

```python
diab = load_diabetes()                        # 442 samples, 10 features -> REGRESSION
Xr, yr = pd.DataFrame(diab.data, columns=diab.feature_names), pd.Series(diab.target)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=0)

wine = load_wine()                            # 178 samples, 13 features, 3 classes -> CLASSIFICATION
Xc, yc = pd.DataFrame(wine.data, columns=wine.feature_names), pd.Series(wine.target)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2,
                                              random_state=0, stratify=yc)
```

| | Diabetes | Wine |
|---|---|---|
| Task | Regression | 3-class classification |
| Train / test | 353 / 89 | 142 / **36** |
| Features | 10 (all pre-standardised) | 13 (wildly different scales) |
| Response | Disease progression, sd ≈ **77.1** | class_0 (59), class_1 (71), class_2 (48) |

> ⚠️ **The wine test set has 36 observations.** One misclassification moves accuracy by **2.8 percentage points**. Almost every "difference" you're about to see in the wine results is one or two samples. Read the rankings as suggestive, not decisive — and say so in your write-up, because noticing it is worth more than the number itself.
>
> **No scaling anywhere.** Unlike Weeks 3 and 4, this is correct: trees split on *thresholds within a single feature at a time*, so any monotone rescaling of a feature produces the identical tree. Wine's `proline` runs into the hundreds and `hue` sits near 1, and it makes no difference whatsoever. **Trees are scale-invariant** — one of their real practical advantages.

### Tutorial baselines

| Model | Result |
|---|---|
| Regression tree, `max_depth=3` | Test RMSE **68.34**, root split on **`s5` < 0.0217** |
| Classification tree, `max_depth=3` | Test accuracy **0.806**, first split on **`proline` < 900.5** |

## Exercise 1 — Tuning tree depth

```python
rows = []
for d in [1, 2, 3, 5, 10, None]:
    m = DecisionTreeRegressor(max_depth=d, random_state=0).fit(Xr_tr, yr_tr)
    rows.append({
        'depth': str(d), 'leaves': m.get_n_leaves(),
        'train': np.sqrt(mean_squared_error(yr_tr, m.predict(Xr_tr))),
        'test':  np.sqrt(mean_squared_error(yr_te, m.predict(Xr_te)))})
res = pd.DataFrame(rows)

plt.plot(res.depth, res.train, 'o-', label='Train RMSE')
plt.plot(res.depth, res.test,  'o-', label='Test RMSE')
plt.xlabel('max_depth'); plt.ylabel('RMSE'); plt.legend(); plt.show()
```

**Tasks 1–2 — the numbers:**

| `max_depth` | 1 | 2 | 3 | **5** | 10 | None |
|---|---|---|---|---|---|---|
| **Train RMSE** | 64.27 | 55.84 | 51.37 | 42.81 | 16.83 | **0.00** |
| **Test RMSE** | 71.50 | 69.03 | 68.34 | **68.17** 🏆 | 83.46 | 83.02 |
| Leaves | 2 | 4 | 8 | 29 | 207 | **344** |

**Task 3 — best test depth: 5**, at RMSE 68.17.

**The two curves behave exactly as the theory says.** Train RMSE falls monotonically and reaches **exactly zero**. Test RMSE falls gently to a shallow minimum at depth 5, then jumps by 15 points. **That's the U-shape**, for the fourth week running.

**Task 4 — what happens at `max_depth=None`, and what is it called?**

**Overfitting.** With no depth limit the tree grows **344 leaves for 353 training observations** — very nearly one leaf per data point. It has memorised the training set, achieving **train RMSE = 0.00** while test RMSE degrades to **83.02**.

> 🔍 **Three things worth saying that the exercise doesn't ask for.**
>
> **1. Predicting the training mean for everyone scores 71.66.** The depth-1 tree scores **71.50**. So a one-split tree is *statistically indistinguishable from the null model*, and even the best tree (68.17) beats the null model by only 5%. **A single tree is a weak model on this data** — which sets up the rest of the lab beautifully: the forest gets to 60.51, a real improvement.
>
> **2. The unpruned tree is worse than predicting the mean** (83.02 vs 71.66). Overfitting doesn't just fail to help — it actively destroys a model that would otherwise have been merely useless.
>
> **3. Depth 3 → 5 buys 0.17 RMSE for 21 extra leaves.** On an 89-observation test set that is pure noise. **The defensible answer is depth 3**, on parsimony — and if the exam asks you to pick, say *why*: the curve is flat between 3 and 5, and a flat region means "choose the simplest". This is the one-standard-error rule from Week 3.
>
> **4. `max_depth` isn't the tuning parameter the lecture recommends.** Slide 14 says grow big, then prune with $\alpha$. Depth-limiting is *early stopping* — cruder, because it forces every branch to stop at the same depth whether or not it deserved to. Exercise 2 does it properly.

## Exercise 2 — Cost-complexity pruning

```python
for a in [0, 0.001, 0.005, 0.01, 0.02, 0.05]:
    m = DecisionTreeClassifier(ccp_alpha=a, random_state=0).fit(Xc_tr, yc_tr)
    print(a, round(accuracy_score(yc_te, m.predict(Xc_te)), 4), m.get_n_leaves())
```

**Tasks 1–2 — the numbers:**

| `ccp_alpha` | 0 | 0.001 | 0.005 | 0.01 | **0.02** | 0.05 |
|---|---|---|---|---|---|---|
| **Test accuracy** | 0.9444 | 0.9444 | 0.9444 | 0.9444 | **0.9444** | 0.8056 |
| Train accuracy | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9718 | 0.9437 |
| **Leaves** | 9 | 9 | 9 | 9 | **5** | 4 |
| Depth | 5 | 5 | 5 | 5 | 4 | 3 |

**Task 3 — which `ccp_alpha` is best?** Five of the six tie at 0.9444. **Break the tie on simplicity: `ccp_alpha = 0.02`**, which achieves the same accuracy with **5 leaves instead of 9**.

**Task 4 — how many leaves does the best pruned tree have? Five.**

> 🔍 **The first four columns are the same tree. Here's how to prove it.**
>
> ```python
> path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(Xc_tr, yc_tr)
> print(np.round(path.ccp_alphas, 4))
> # [0.  0.0135  0.0137  0.0338  0.0843  0.2207  0.2648]
> ```
>
> These are the **effective alphas** — the only values at which the tree actually changes. **Every $\alpha$ below 0.0135 prunes nothing at all**, which is why 0, 0.001, 0.005 and 0.01 give byte-identical 9-leaf trees. The lab's grid wastes four of its six points in a dead zone. **Real tuning uses `cost_complexity_pruning_path` to generate the grid**, rather than guessing round numbers — that's the single most useful thing to take from this exercise.
>
> **Watch the train accuracy column to see pruning working.** It sits at a suspicious **1.0000** for the unpruned tree — 142 training points classified perfectly — and drops to 0.9718 at $\alpha = 0.02$ **while test accuracy holds**. That gap closing with no cost is the definition of removing overfitting. At $\alpha = 0.05$ you've cut into muscle: one more leaf gone and test accuracy collapses by 14 points.

> ⚠️ **Two caveats you should state in your write-up.**
>
> **1. Choosing $\alpha$ on the test set is the Week 3 sin.** Slide 14 says choose it by cross-validation *on the training data*. Doing it properly:
>
> | $\alpha$ | 0 | 0.001 | 0.005 | 0.01 | **0.02** | 0.05 |
> |---|---|---|---|---|---|---|
> | 5-fold CV accuracy (train only) | 0.9291 | 0.9291 | 0.9291 | 0.9291 | **0.9293** | 0.9227 |
>
> Same winner, honestly obtained — though by a margin of 0.0002, which is nothing. Use `StratifiedKFold` here, since it's classification.
>
> **2. A plain depth-4 tree beats every pruned tree**, scoring **0.9722** against 0.9444. The pruning grid never finds it. Don't conclude that pruning is inferior — conclude that **a coarse grid over the wrong parameter can miss the good region entirely**, and that on 36 test samples the whole ranking rests on one wine.

## Exercise 3 — Tuning a random forest

```python
rf = RandomForestClassifier(random_state=0).fit(Xc_tr, yc_tr)
print(accuracy_score(yc_te, rf.predict(Xc_te)))                       # 1.0

for mf in [1, 3, 6, 13, 'sqrt']:
    m = RandomForestClassifier(max_features=mf, random_state=0).fit(Xc_tr, yc_tr)
    print(mf, accuracy_score(yc_te, m.predict(Xc_te)))

imp = pd.Series(rf.feature_importances_, index=Xc.columns).sort_values()
imp.plot(kind='barh'); plt.show()
```

**Task 1 — default random forest test accuracy: 1.000. Every wine in the test set classified correctly.**

**Task 2 — `max_features`:**

| `max_features` | 1 | 3 | 6 | **13** | 'sqrt' (= 3) |
|---|---|---|---|---|---|
| **Test accuracy** | 1.000 | 1.000 | 1.000 | **0.9722** ⚠️ | 1.000 |
| 5-fold CV on train | **0.9860** 🏆 | 0.9719 | 0.9645 | 0.9645 | 0.9719 |

**`max_features = 13` is the only setting that gets anything wrong** — and 13 is $p$, which means **that row is bagging, not a random forest**. Every genuinely decorrelated setting is perfect.

> 🔍 **This is the cleanest empirical demonstration of Part 7 you'll get.** The theory says restricting each split to $m < p$ predictors decorrelates the trees and lowers the variance floor. Here the only configuration that fails is the one where the restriction is switched off. And on cross-validation — which has 142 samples' worth of evidence instead of 36 — the ordering is monotone and unambiguous: **the fewer features each split may consider, the better the forest performs** (0.9860 at $m=1$, down to 0.9645 at $m=13$).
>
> **$m = 1$ winning outright is worth pausing on.** Each split is handed *one random predictor* and must use it — the individual trees are close to random and clearly worse than a properly grown tree. The ensemble is still the best of the lot. **Individual weakness plus maximum diversity beats individual strength plus uniformity.** Note also that `sqrt` here means $\lfloor\sqrt{13}\rfloor = 3$, matching the slides' "$m \approx \sqrt{p}$, e.g. 4 of 13" rule.

**Task 3 — feature importances (top 5):**

| Feature | Importance |
|---|---|
| **proline** | **0.191** |
| color_intensity | 0.176 |
| flavanoids | 0.170 |
| alcohol | 0.129 |
| od280/od315_of_diluted_wines | 0.100 |

**The most important feature for wine variety is `proline`** — which agrees with the single tree, whose very first split was `proline < 900.5`. The bottom five features (`ash`, `nonflavanoid_phenols`, `proanthocyanins`, `alcalinity_of_ash`, `magnesium`) contribute under 3% between them.

> **Two honest qualifications.** The top three are separated by 0.02 on a 142-row training set — treat them as a **tied leading group**, not a ranking. And `flavanoids`, `total_phenols` and `od280` are chemically related measurements, so their shared signal is being **split three ways** (the correlation caveat from Part 7). The 13 importances sum to 1 by construction, so they are shares of a fixed pie, not absolute measures of usefulness.

**Task 4 — single tree vs forest:**

| Model | Wine test accuracy |
|---|---|
| Single tree (pruned, $\alpha = 0.02$) | 0.9444 |
| Single tree (best depth = 4) | 0.9722 |
| **Random forest** | **1.000** 🏆 |

The forest wins, as expected — but on a 36-sample test set the gap over the best single tree is **exactly one wine**. Report it with that caveat.

## Exercise 4 — Comparing all methods

```python
gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                 random_state=0).fit(Xc_tr, yc_tr)
print(accuracy_score(yc_te, gbc.predict(Xc_te)))     # 0.9722
```

**Task 2 — the comparison table:**

| Method | Wine accuracy | Diabetes RMSE |
|---|---|---|
| Single Tree (best depth/alpha) | 0.944 | 68.34 |
| **Random Forest** | **1.000** 🏆 | **60.51** 🏆 |
| Gradient Boosting | 0.972 | 65.54 |

**Random forest wins both tasks.** Boosting comes second on both.

**Task 3 — the key difference between bagging and boosting:**

**Bagging builds trees independently and in parallel, each on a bootstrap resample of the data, then averages them — it reduces variance.** **Boosting builds trees sequentially, each one fitted to the residual errors left by the trees before it, adding a shrunken contribution each time — it reduces bias.** Bagging's trees are deep and interchangeable; boosting's are small and order-dependent. More trees never hurts bagging; more trees eventually overfits boosting.

> 🔍 **The most instructive result in the lab: boosting lost, and the summary table says it shouldn't have.**
>
> The notebook's own summary calls gradient boosting the "highest accuracy" method. It came second on both datasets. **This is not a contradiction — it's boosting's headline weakness ("needs careful tuning") showing up live.**
>
> The tutorial cell hands you the diagnosis, if you read the plot it draws:
>
> ```python
> rmse_staged = [np.sqrt(mean_squared_error(yr_te, p)) for p in gb.staged_predict(Xr_te)]
> print(min(rmse_staged), np.argmin(rmse_staged) + 1)
> # 60.43 at 13 trees   (vs 65.54 at the full 200)
> ```
>
> **The boosted model reaches RMSE 60.43 after 13 trees and then gets steadily worse for the remaining 187.** At `learning_rate=0.1` with `max_depth=3` on 353 rows, it fits the signal almost immediately and spends the rest of its budget fitting noise. The final score of 65.54 is a model that was, briefly, the best thing in the lab.
>
> **Fix it the way Part 8 says — learn slower:**
>
> | Boosting configuration | Diabetes test RMSE |
> |---|---|
> | Lab default: 200 trees, lr = 0.1, depth 3 | 65.54 |
> | Stopped at its own optimum (13 trees) | 60.43 |
> | Stumps: 200 trees, lr = 0.05, **depth 1** | 59.56 |
> | **500 trees, lr = 0.01, depth 2** | **59.02** 🏆 |
>
> **Properly tuned, boosting beats the random forest after all** (59.02 vs 60.51) — and the winning configuration is the slowest, smallest one. Small learning rate, shallow trees, many of them: exactly the recipe in Part 8. **The lecture's ranking is right; the lab's default hyperparameters are wrong.** Saying this in a write-up demonstrates you understand *why* boosting is fragile, rather than just reciting that it is.
>
> **Contrast with the random forest**, which was excellent at its untouched defaults on both datasets. That robustness is the real practical argument for forests: **a random forest out of the box is usually near its ceiling; boosting out of the box is anywhere at all.**

---

# Key takeaways (as the slides state them)

1. **Decision trees:** recursive binary splits, easy to interpret, **prone to overfitting**.
2. **Bagging:** average over many bootstrap trees; **reduces variance**.
3. **Random forests:** like bagging, but only $m \approx \sqrt{p}$ features considered at each split; **decorrelates the trees**.
4. **Boosting:** sequential trees fitted to residuals with a learning rate; a slow learner, often highly accurate.
5. **XGBoost / LightGBM:** fast, regularised boosting implementations.

---

# Formula sheet

| Concept | Formula |
|---|---|
| Regression tree objective | $\sum_{j=1}^{J}\sum_{i\in R_j}(y_i - \hat{y}_{R_j})^2$ |
| Leaf prediction (regression) | $\hat{y}_{R_j} = \dfrac{1}{n_j}\sum_{i \in R_j} y_i$ |
| Leaf prediction (classification) | Majority class in $R_j$ |
| Cost-complexity criterion | $C_\alpha(T) = \sum_{m=1}^{|T|}\sum_{i:x_i\in R_m}(y_i-\hat{y}_{R_m})^2 + \alpha|T|$ |
| Gini index | $G = \sum_{k=1}^{K}\hat{p}_{mk}(1-\hat{p}_{mk})$ |
| Cross-entropy | $D = -\sum_{k=1}^{K}\hat{p}_{mk}\log\hat{p}_{mk}$ |
| Classification error rate | $E = 1 - \max_k \hat{p}_{mk}$ |
| Weighted child impurity | $\frac{n_L}{n}G_L + \frac{n_R}{n}G_R$ |
| Candidate splits per node | $p \times (n-1)$ |
| Random forest split subset | $m \approx \sqrt{p}$ (classification), $m \approx p/3$ (regression) |
| Bagging = RF with | $m = p$ |
| Variance of a correlated average | $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ |
| RMSE | $\sqrt{\frac{1}{n}\sum(y_i-\hat{y}_i)^2}$ |

---

# sklearn cheat sheet

| Task | Code |
|---|---|
| Regression tree | `DecisionTreeRegressor(max_depth=3, random_state=0)` |
| Classification tree | `DecisionTreeClassifier(max_depth=3, random_state=0)` |
| **Visualise a tree** | `plot_tree(model, feature_names=X.columns, filled=True, rounded=True)` |
| Number of leaves | `model.get_n_leaves()` |
| Depth of fitted tree | `model.get_depth()` |
| Root split feature | `X.columns[model.tree_.feature[0]]` |
| Root split threshold | `model.tree_.threshold[0]` |
| Cost-complexity pruning | `DecisionTreeClassifier(ccp_alpha=0.02)` |
| **Valid alphas for the grid** | `model.cost_complexity_pruning_path(X, y).ccp_alphas` |
| Splitting criterion | `criterion='gini'` (default) or `'entropy'` / `'squared_error'` |
| Random forest | `RandomForestRegressor(n_estimators=200, max_features='sqrt')` |
| Bagging (via RF) | `RandomForestRegressor(max_features=None)` |
| Feature importances | `pd.Series(m.feature_importances_, index=X.columns).sort_values()` |
| Out-of-bag score | `RandomForestClassifier(oob_score=True).oob_score_` |
| Gradient boosting | `GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3)` |
| **Boosting error vs #trees** | `[metric(y_te, p) for p in gb.staged_predict(X_te)]` |
| Cross-validation | `cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))` |
| RMSE | `np.sqrt(mean_squared_error(y_true, y_pred))` |

> **No `StandardScaler` anywhere.** Trees are invariant to monotone rescaling of individual features, so scaling is unnecessary — the first week since Week 2 where that's true.

### Six mistakes that cost marks

1. **Pruning the trees inside a bagged model or random forest.** Ensemble members are grown deep and unpruned on purpose — the averaging handles the variance, and pruning would add bias that averaging can't remove.
2. **Saying random forests reduce bias.** They reduce **variance**. Boosting is the one that primarily reduces bias.
3. **Choosing depth, $\alpha$, or `max_features` on the test set.** Use cross-validation on the training set; the test set is scored once, at the end.
4. **Forgetting to weight child impurities by node size** in a Gini calculation.
5. **Assuming more trees is always safe.** True for bagging and random forests, **false for boosting** — the lab's boosted model peaks at 13 trees and degrades for the next 187.
6. **Reading feature importance as a coefficient.** It's unsigned, biased toward continuous and high-cardinality predictors, and gets diluted across correlated features.

---

# Quick self-test

Cover the answers.

1. **Why is recursive binary splitting called "greedy", and what does it cost you?** — *At each step it takes the split that most reduces RSS right now, with no lookahead. A worse split now might enable two excellent splits later, and the algorithm will never find that. We accept it because searching all possible partitions is computationally infeasible.*
2. **Why grow a large tree and prune it, instead of stopping early when improvement gets small?** — *Because a split that looks worthless can enable a very valuable split beneath it. Early stopping is greedy about stopping as well as splitting; growing first lets you see the whole tree before cutting.*
3. **What does $\alpha$ do in $C_\alpha(T) = RSS + \alpha|T|$, and how do you choose it?** — *It prices each extra leaf. $\alpha = 0$ keeps the full tree; larger $\alpha$ prunes more. Choose it by K-fold cross-validation, then refit on the full training data at the chosen value. It's structurally identical to $\lambda$ in Ridge/Lasso.*
4. **Why isn't the classification error rate used for growing a tree?** — *It's too crude: it only registers a change when the majority class changes, so it ignores splits that make a node genuinely purer. Gini and cross-entropy are sensitive to every shift in the class proportions. Error rate is still preferable for pruning.*
5. **A node has 8 Class A and 2 Class B. Compute the Gini index.** — *$G = 0.8 \times 0.2 + 0.2 \times 0.8 = 0.32$. Zero means pure; 0.5 is the two-class maximum.*
6. **Why do random forests deliberately hide predictors from each split?** — *Because with one dominant predictor, every bagged tree splits on it first and the trees become near-copies. The variance of an average of correlated quantities has a floor of $\rho\sigma^2$ that no number of trees removes. Restricting to $m$ random predictors lowers $\rho$: each tree is worse, the ensemble is better.*
7. **When is a random forest exactly bagging?** — *When $m = p$, i.e. `max_features=None`. In the lab this is the only wine setting that misclassified anything.*
8. **Your boosted model's test error rises after 40 trees. What's wrong and what do you change?** — *Boosting is overfitting, because unlike bagging it keeps fitting residuals — including noise. Reduce `n_estimators` (or use early stopping), lower the `learning_rate` and add trees to compensate, and shrink `max_depth`. In the lab, dropping lr from 0.1 to 0.01 with depth 2 took RMSE from 65.54 to 59.02.*
9. **Do you need to standardise features for a tree? For a random forest?** — *No, for either. Splits are thresholds on one feature at a time, so any monotone rescaling gives an identical tree. Contrast with KNN and regularised logistic regression in Week 4, where scaling is mandatory.*
10. **You fit a depth-unlimited regression tree: train RMSE 0.00, test RMSE 83, and predicting the training mean scores 71.7. Diagnose it.** — *Severe overfitting. With 344 leaves for 353 observations it has memorised the training set, and it is now worse than the null model. Limit the depth or prune with cross-validated $\alpha$ — or better, average many such trees in a random forest, which took the same data to RMSE 60.5.*
