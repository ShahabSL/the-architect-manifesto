# Mental Models for Machine Learning

## 1. The Geometric Interpretation of the Dot Product

The dot product of two vectors (A and B) is a scalar value that represents the projection of vector A onto vector B, scaled by the magnitude of B. It effectively measures how much one vector "goes in the direction of" another.

-   **Positive Result:** The angle between the vectors is less than 90 degrees (acute). The vectors point in a generally similar direction.
-   **Negative Result:** The angle between the vectors is greater than 90 degrees (obtuse). The vectors point in generally opposite directions.
-   **Zero Result:** The angle between the vectors is exactly 90 degrees (orthogonal). The vectors are perpendicular and have no directional relationship.

## 2. The Practical Meaning of a Gradient in Machine Learning

The gradient is a vector containing the partial derivatives of the loss function with respect to each of the model's parameters (weights and biases).

In practice, each number in the gradient vector tells us two things about a specific parameter:
1.  **The Direction:** The sign of the number tells us whether increasing the parameter will increase or decrease the loss.
2.  **The Magnitude:** The value of the number tells us how sensitive the loss is to that parameter. A larger value means a small change to that parameter will cause a large change in the loss.

During training, we move in the **opposite direction** of the gradient (gradient descent) to find the steepest path toward a lower loss.

## 3. Entropy vs. Cross-Entropy

**Entropy** measures the amount of uncertainty or "surprise" inherent in a single probability distribution.
-   **Low Entropy:** The distribution has one or very few highly probable outcomes (e.g., a loaded die). There is low uncertainty.
-   **High Entropy:** The outcomes in the distribution are closer to being equally probable (e.g., a fair coin flip). There is high uncertainty.

**Cross-Entropy** is a concept that involves two probability distributions: the "true" distribution (`p`) and a predicted distribution (`q`) from our model. It measures the average number of bits needed to identify an event from `p` when your coding scheme is optimized for `q`.

In machine learning, it serves as a loss function. It's preferred for classification because:
-   It's a measure of the "distance" between the model's predicted probabilities and the actual class labels.
-   When used with a softmax activation function, it creates a convex loss landscape, making the optimization process (gradient descent) much more reliable and less likely to get stuck in local minima compared to alternatives like Mean Squared Error.

