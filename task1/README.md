# Task Description
According to the United Nations, one in three people worldwide do not have access to safe drinking water. Unsafe water is a leading risk factor for death, especially at low incomes, and is one of the world's largest health and environmental problems. Groundwater pollution occurs when pollutants are released into the ground and make their way down into groundwater. While water contamination can occur from naturally occurring contaminants, such as arsenic or fluoride, common causes of water pollution are on-site sanitation systems, effluent from wastewater treatment plants, petrol filling stations or agricultural fertilizers.

In order to prevent outbreaks and incidents of water poisonings, detecting ground-water contamination is crucial. Geostatistics has often utilized the Gaussian Process (GP) to model the spatial pattern of pollutant concentrations in the ground. Usually, a data point in 2D represents a geological well where a sample was taken from a bore hole to measure concentration of pollutants.

In the following task, Gaussian Process regression (or a similar method) will be used in order to model groundwater pollution, and try to predict the concentration of pollutants at previously unmeasured wells (points). 

# Challenges
We envisage that in order to solve this task you need to overcome three challenges - each requiring a specific strategy.
1. Model selection - You will need to find the right kernel and its hyper-parameters that model the GP faithfully. With Bayesian models, a commonly used principle in choosing the right kernel or hyper-parameters is to use the "data likelihood", otherwise known as the marginal likelihood to find the best model.
2. Large scale learning - Natively, GP inference is computationally intensive for large datasets and common-place computers. The inference requires 
O(N³) basic operations in order find the posterior distributions. For large datasets this becomes infeasible. In order to solve this problem, low-rank approximations will be used. The most popular are the Nyström method, using random features and/or other scalable clustering-based approaches.
3. Asymmetric costs: We utilize a specifically designed cost function, where deviation from the true concentration levels is penalized, and you are rewarded for correctly predicting safe regions. Under this specific cost function, the mean prediction might not be optimal. Note that the mean prediction refers to the optimal decision with respect to a general squared loss and some posterior distribution over the true value to be predicted.


# Metrics
\mathcal{L}_C(f(x), \hat{f}(x)) =
    \begin{cases}
        1 \times (f(x) - \hat{f}(x))^2  \enspace & \text{if } f(x) > \theta, \hat{f}(x) \ge f(x) \text{ or } f(x) \le \theta, \hat{f}(x) > f(x), \\
        20 \times (f(x) - \hat{f}(x))^2  & \text{if } f(x) > \theta, f(x) \geq \hat{f}(x) \ge \theta \text{ or } f(x) \le \theta, \hat{f}(x) \le f(x), \\
        100 \times (f(x) - \hat{f}(x))^2 & \text{if } f(x) > \theta, \hat{f}(x) < \theta.
    \end{cases}
