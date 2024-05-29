# Missing Data

First we should be able to talk and distinguish between different types of missing data
Suppose we have independent and identically distributed data points $(X_1, \ldots, X_n\)$, where $\(X_i = (X_{i1}, \ldots, X_{ip})\)$.
For each $\(i \in [n]\)$, we may not observe all the coordinates of $\(X_i\)$ in that there exists random Bernoulli
missingness indicators $\(M_{i1}, \ldots, M_{ip} \in \{0, 1\}\)$ (not necessarily independent) and we observe only
$\(\{X_{ij} : M_{ij} = 0, \text{ for } j \in [p]\}\).$

There are three levels of assumptions regarding the missingness indicators $\(M_i \in \{0, 1\}^p\)$:

$$
\begin{aligned}
    &\text{1. Missing completely at random (MCAR), where } M_i \text{ is independent of } X_i. \\
    &\text{2. Missing at random (MAR), where } M_i \text{ can depend only on } X_i \text{ only through the observed entries } \{X_{ij} : M_{ij} = 0\}. \\
    &\text{3. Missing not at random (MNAR), where } M_i \text{ can depend on } X_i \text{ in arbitrary ways}.
\end{aligned}
$$

**Examples of MCAR**

$$
\begin{aligned}
    &\text{1. For each } i, \text{ we choose exactly one coordinate } j \in [p] \text{ and set } M_{ij} = 1. \text{ We set all other coordinates of } M_i \text{ to } 0. \\
    &\text{2. Each } M_{i1}, \ldots, M_{ip} \text{ are independent and that } M_{ij} \sim \text{ Ber}(\theta_j).
\end{aligned}
$$

**Examples of MAR**

$$
\begin{aligned}
    &\text{1. If } X_{i1} > 0, \text{ then we set } M_{i2} = 1 \text{ and all other coordinates of } M_i \text{ to be zero.} \\
    &\text{2. The } M_{ij} \text{ are independent given } X_i \text{ and } P(M_{ij} \mid X_i) = f_j(X_{i1}).
\end{aligned}
$$

More simply,
MCAR : No pattern in the missing features. (very rare in practice)

MAR: Missing features are dependent on other observed values.

MNAR:Missing features are dependent on the true values.(eg: salary missing from high-earners)

With that said, now we can deal with the question:

You are given a dataset for modeling. You've found it has a lot of missing feature values - how will you proceed?

Ans:
First I would ask few clarification questions
- What is the data generation/gathering process - is there a bug and can easily be rectified to get the full data.
- Any constraints on the models to use - (Tree based models have a very natural way of dealing with missing data).
- How much data is missing ?

Then roughly i'll try to cover different approaches

Handling missing data is impossible under MNAR, difficult under MAR, and sometimes feasible under MCAR. It is possible to test for MCAR, but not MAR. Some heuristics used in practice are:
1. Deletion: remove the rows or entire columns depending on the situation
	   `cons`: Can introduce bias(only valid under MCAR) and drop performance.
2. Simple Imputation : replace with Mean/Median/Mode. For time series simple interpolation.
	  `cons`: Underestimates the variance of the feature and renders any confidence intervals/hypothesis testing invalid. 
3. Model based imputation: Under MAR, we should be able to approximate missing values from the other features. Thus we have two options here
	1. `model conditional distributions` via regression or classification methods applied onto data points where we have no missing features. For example, use regression to get the estimate missing value additionally we can include euclidean distance to weigh the samples during training.
	2. `Estimate the joint distribution of the data` - Generative models have no problem dealing with missing data. We can use those to impute and then later switch to a discriminative model for better results.
In practice, the best approach is determined by trail and error, we'll use cross-validation to determine the best method. In practice ,i've found it works best to impute but also introduce missingness indicator as additional feature.
