# %% [markdown]
# # ML HW1
#

# %%
import numpy as np
import pandas as pd

data_x_df = pd.read_csv('X.csv')
data_t_df = pd.read_csv('T.csv')
data_df = pd.concat([data_x_df, data_t_df], axis=1)
data_df

# %%
dataset = np.array(data_df)
np.random.shuffle(dataset)

train_ratio = 0.8
train_size = int(len(dataset) * train_ratio)

train_feature = dataset[:train_size, :-1]
test_feature = dataset[train_size:, :-1]
train_target = dataset[:train_size, -1]
test_target = dataset[train_size:, -1]

# %% [markdown]
# ## 1. Feature Selection
#


# %%
def optimal_weight_ml(phi, target):
    return np.linalg.pinv(phi) @ target


# %% [markdown]
# ### (a)

# %% [markdown]
# ### M = 1
#
# $$ y(\textbf{x}, \textbf{w}) = \textit{w}_{0} + \sum^{D}_{\textit{i} = 1} \textit{w}_{i}\textit{x}_{i} = \textbf{w}^{T} \phi(x) $$
# where
# $$ \textbf{w} = \begin{pmatrix} \textit{w}_{0} & \textit{w}_{1} & ... & \textit{w}_{D} \end{pmatrix}^{T} $$
# $$ \phi(x) = \begin{pmatrix} 1 & \textit{x}_{1} & ... & \textit{x}_{D} \end{pmatrix}^{T} $$
#


# %%
def phi_m1(features):
    """Computes phi matrix with m = 1.
    
    Args:
        features: np.ndarray with shape (n_samples, n_features)
        
    Returns:
        phi: np.ndarray with shape (n_samples, 1 + n_features)
    """
    m_0 = np.ones((features.shape[0], 1))  # 1
    m_1 = features  # x_1, x_2, ..., x_D
    return np.concatenate((m_0, m_1), axis=-1)


# %%
train_phi_m1 = phi_m1(train_feature)
test_phi_m1 = phi_m1(test_feature)

w_m1 = optimal_weight_ml(train_phi_m1, train_target)

# %% [markdown]
# ### M = 2
#
# $$ y(\textbf{x}, \textbf{w}) = \textit{w}_{0} + \sum^{D}_{\textit{i} = 1} \textit{w}_{i}\textit{x}_{i} + \sum^{D}_{\textit{i} = 1}\sum^{D}_{\textit{j} = 1} \textit{w}_{ij}\textit{x}_{i}\textit{x}_{j} = \textbf{w}^{T} \phi(x) $$
# where
# $$ \textbf{w} = \begin{pmatrix} \textit{w}_{0} & \textit{w}_{1} & ... & \textit{w}_{D} & \textit{w}_{11} & \textit{w}_{12} & ... & \textit{w}_{DD} \end{pmatrix}^{T} $$
# $$ \phi(x) = \begin{pmatrix} 1 & \textit{x}_{1} & ... & \textit{x}_{D} & \textit{x}_{1}\textit{x}_{1} & \textit{x}_{1}\textit{x}_{2} & ... & \textit{x}_{D} \textit{x}_{D} \end{pmatrix}^{T} $$
#


# %%
def phi_m2(features):
    """Computes phi matrix with m = 2.
    
    Args:
        features: np.ndarray with shape (n_samples, n_features)
        
    Returns:
        phi: np.ndarray with shape (n_samples, 1 + n_features + n_features ** 2)
    """
    m_0 = np.ones((features.shape[0], 1))  # 1
    m_1 = features  # x_1, x_2, ..., x_D

    # x_1^2, x_1x_2, ..., x_1x_D, x_2^2, x_2x_3, ..., x_2x_D, ..., x_D^2
    m_2 = np.expand_dims(features, axis=-1) @ np.expand_dims(features, axis=-2)
    m_2 = m_2.reshape(m_2.shape[0], -1)
    return np.concatenate((m_0, m_1, m_2), axis=-1)


# %%
train_phi_m2 = phi_m2(train_feature)
test_phi_m2 = phi_m2(test_feature)

w_m2 = optimal_weight_ml(train_phi_m2, train_target)

# %% [markdown]
# ### Root Mean Square Error
#


# %%
def root_mean_square_error(y, t):
    return np.sqrt(np.mean((y - t)**2))


train_rms_m1 = root_mean_square_error(train_phi_m1 @ w_m1, train_target)
test_rms_m1 = root_mean_square_error(test_phi_m1 @ w_m1, test_target)

train_rms_m2 = root_mean_square_error(train_phi_m2 @ w_m2, train_target)
test_rms_m2 = root_mean_square_error(test_phi_m2 @ w_m2, test_target)

# %%
print(f'train rms m1: {train_rms_m1}')
print(f'test rms m1: {test_rms_m1}')
print(f'train rms m2: {train_rms_m2}')
print(f'test rms m2: {test_rms_m2}')

# %% [markdown]
# ### (b)

# %% [markdown]
# ### Weight Analysis of M = 1
#
# I used the weight of the model to analyze the importance of each feature. The weight of the model is the coefficient of each feature. The larger the weight is, the more important the feature is.
#
# > The weight with index 0 is not considered as a feature because it is the bias of the model.
#

# %%
print(f'w_m1: {w_m1}')
abs_importance = np.abs(w_m1[1:])  # index #0 is bias
importance_rank = np.argsort(abs_importance)[::-1]
print(f'Rank of features\' contribution: {importance_rank}')
print(f'The most important feature: #{importance_rank[0]} {data_df.columns[importance_rank[0]]}')

# %%
for i in range(11):
    weight_without_i_feature = np.copy(w_m1)
    weight_without_i_feature[i + 1] = 0  # index #0 is bias

    train_rms_without_i_feature = root_mean_square_error(train_phi_m1 @ weight_without_i_feature,
                                                         train_target)
    test_rms_without_i_feature = root_mean_square_error(test_phi_m1 @ weight_without_i_feature,
                                                        test_target)

    print(f'Remove feature #{i}:')
    print(f'train rms m1: {train_rms_without_i_feature}, test rms m1: {test_rms_without_i_feature}')

# %% [markdown]
# Without feature #7 (density), the rms error is larger than the model without other features, which means that feature #7 is the most important feature in the model.
#

# %% [markdown]
# ## 2. Maximum Likelihood Approach
#

# %% [markdown]
# ### (a)

# %% [markdown]
# Use Gaussian basis, sigmoid basis and hybrid basis which is the combination of first and second order polynomial, Gaussian and sigmoid basis.
#
# - Gaussian basis: The output is larger when the input is closer to the mean of the basis. It is suitable for get rid of extreme values.
# - Sigmoid basis: The output remains the same order of the input.

# %% [markdown]
# ### (b)


# %%
def get_mean_and_std(features):
    """Computes the mean and standard deviation of each feature.
    
    Args:
        features: np.ndarray with shape (n_samples, n_features)
        
    Returns:
        mean: np.ndarray with shape (1, n_features)
        std: np.ndarray with shape (1, n_features)
    """
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    return mean, std


def phi_gaussian(features, mean, std):
    """Computes the Gaussian phi.
    
    Args:
        features: np.ndarray with shape (n_samples, n_features)
        mean: np.ndarray with shape (1, n_features)
        std: np.ndarray with shape (1, n_features)
        
    Returns:
        phi: np.ndarray with shape (n_samples, 1 + n_features)
    """
    m_0 = np.ones((features.shape[0], 1))  # 1
    gaussians = np.exp(-((features - mean) / std)**2 / 2)
    return np.concatenate((m_0, gaussians), axis=-1)


# %%
train_mean, train_std = get_mean_and_std(train_feature)
train_phi_gaussian = phi_gaussian(train_feature, train_mean, train_std)
test_phi_gaussian = phi_gaussian(test_feature, train_mean, train_std)

w_gaussian = optimal_weight_ml(train_phi_gaussian, train_target)

# %%
train_rms_gaussian = root_mean_square_error(train_phi_gaussian @ w_gaussian, train_target)
test_rms_gaussian = root_mean_square_error(test_phi_gaussian @ w_gaussian, test_target)

print(f'train rms gaussian: {train_rms_gaussian}')
print(f'test rms gaussian: {test_rms_gaussian}')


# %%
def phi_sigmoid(features, mean, std):
    """Computes the sigmoid phi.
    
    Args:
        features: np.ndarray with shape (n_samples, n_features)
        mean: np.ndarray with shape (1, n_features)
        std: np.ndarray with shape (1, n_features)
        
    Returns:
        phi: np.ndarray with shape (n_samples, 1 + n_features)
    """
    m_0 = np.ones((features.shape[0], 1))  # 1
    sigmoids = 1 / (1 + np.exp(-(features - mean) / std))
    return np.concatenate((m_0, sigmoids), axis=-1)


# %%
train_phi_sigmoid = phi_sigmoid(train_feature, train_mean, train_std)
test_phi_sigmoid = phi_sigmoid(test_feature, train_mean, train_std)

w_sigmoid = optimal_weight_ml(train_phi_sigmoid, train_target)

# %%
train_rms_sigmoid = root_mean_square_error(train_phi_sigmoid @ w_sigmoid, train_target)
test_rms_sigmoid = root_mean_square_error(test_phi_sigmoid @ w_sigmoid, test_target)

print(f'train rms sigmoid: {train_rms_sigmoid}')
print(f'test rms sigmoid: {test_rms_sigmoid}')


# %%
def phi_hybrid(features, mean, std):
    """Computes the hybrid phi.
    
    Args:
        features: np.ndarray with shape (n_samples, n_features)
        mean: np.ndarray with shape (1, n_features)
        std: np.ndarray with shape (1, n_features)
        
    Returns:
        phi: np.ndarray with shape (n_samples, 1 + n_features + n_features ** 2 + n_features + n_features)
    """
    m_0 = np.ones((features.shape[0], 1))  # 1
    m_1 = features  # x_1, x_2, ..., x_D
    # x_1^2, x_1x_2, ..., x_1x_D, x_2^2, x_2x_3, ..., x_2x_D, ..., x_D^2
    m_2 = np.expand_dims(features, axis=-1) @ np.expand_dims(features, axis=-2)
    m_2 = m_2.reshape(m_2.shape[0], -1)
    gaussians = np.exp(-((features - mean) / std)**2 / 2)
    sigmoids = 1 / (1 + np.exp(-(features - mean) / std))
    return np.concatenate((m_0, m_1, m_2, gaussians, sigmoids), axis=-1)


# %%
train_phi_hybrid = phi_hybrid(train_feature, train_mean, train_std)
test_phi_hybrid = phi_hybrid(test_feature, train_mean, train_std)

w_hybrid = optimal_weight_ml(train_phi_hybrid, train_target)

# %%
train_rms_hybrid = root_mean_square_error(train_phi_hybrid @ w_hybrid, train_target)
test_rms_hybrid = root_mean_square_error(test_phi_hybrid @ w_hybrid, test_target)

print(f'train rms hybrid: {train_rms_hybrid}')
print(f'test rms hybrid: {test_rms_hybrid}')

# %%
print('Maximum likelihood:')

pd.DataFrame(
    data={
        'train rms': [
            np.mean(train_rms_m1),
            np.mean(train_rms_m2),
            np.mean(train_rms_gaussian),
            np.mean(train_rms_sigmoid),
            np.mean(train_rms_hybrid),
        ],
        'test rms': [
            np.mean(test_rms_m1),
            np.mean(test_rms_m2),
            np.mean(test_rms_gaussian),
            np.mean(test_rms_sigmoid),
            np.mean(test_rms_hybrid),
        ]
    },
    index=[
        'polynomial m=1',
        'polynomial m=2',
        'gaussian',
        'sigmoid',
        'hybrid',
    ],
)

# %% [markdown]
# From the observation, Gaussian basis is the worst in the testing dataset. Other basis have almost similar results. All of the basis except Gaussian basis remain the order of the inputs and outputs. Therefore, we can infer that most features are highly correlated.
#
# Hybrid basis performs the best in the training dataset, but not in the testing dataset, which means that the model is too complex in this case. The model is overfitting, and it is not general enough to predict the testing dataset.

# %% [markdown]
# ### (c)

# %% [markdown]
# ### N-Fold Cross-Validation
#


# %%
def n_fold(data, n_split):
    """Splits the data into n_fold.
    
    Args:
        data: np.ndarray with shape (n_samples, ...)
        n_split: int
        
    Yields:
        train_index: np.ndarray with shape (n_samples * (n_split - 1) // n_split, ...)
        test_index: np.ndarray with shape (n_samples // n_split, ...)
    """
    n_samples = data.shape[0]
    n_samples_per_fold = n_samples // n_split
    for i in range(n_split):
        train_index = np.concatenate((
            np.arange(0, i * n_samples_per_fold),
            np.arange((i + 1) * n_samples_per_fold, n_samples),
        ))
        test_index = np.arange(
            i * n_samples_per_fold,
            (i + 1) * n_samples_per_fold,
        )
        yield train_index, test_index


def fit_ml(train_feature, train_target, test_feature, test_target, mean, std, phi_fn):
    """Fits the phi function.
    
    Args:
        train_feature: np.ndarray with shape (n_samples, n_features)
        train_target: np.ndarray with shape (n_samples, 1)
        test_feature: np.ndarray with shape (n_samples, n_features)
        test_target: np.ndarray with shape (n_samples, 1)
        mean: np.ndarray with shape (1, n_features)
        std: np.ndarray with shape (1, n_features)
        phi_fn: function with signature (features, mean, std) -> phi
        
    Returns:
        train_rms: float
        test_rms: float
        weights: np.ndarray with shape (phi.shape[1], 1)
    """
    train_phi = phi_fn(train_feature, mean, std)
    test_phi = phi_fn(test_feature, mean, std)
    weights = optimal_weight_ml(train_phi, train_target)
    train_rms = root_mean_square_error(train_phi @ weights, train_target)
    test_rms = root_mean_square_error(test_phi @ weights, test_target)
    return train_rms, test_rms, weights


# %% [markdown]
# Use 5-fold to select the best sigmoid model from different std, which are [50%, 75%, 100%, 125%, 150%] of the original std.

# %%
train_rms_std_50_list, test_rms_std_50_list = [], []
train_rms_std_75_list, test_rms_std_75_list = [], []
train_rms_std_100_list, test_rms_std_100_list = [], []
train_rms_std_125_list, test_rms_std_125_list = [], []
train_rms_std_150_list, test_rms_std_150_list = [], []

for i, (train_index, test_index) in enumerate(n_fold(dataset, 5)):
    train_feature, train_target = dataset[train_index, :-1], dataset[train_index, -1]
    test_feature, test_target = dataset[test_index, :-1], dataset[test_index, -1]

    train_mean, train_std = get_mean_and_std(train_feature)

    # 50%
    train_rms_std_50, test_rms_std_50, _ = fit_ml(train_feature, train_target, test_feature,
                                                  test_target, train_mean, train_std * 0.5,
                                                  phi_sigmoid)

    train_rms_std_50_list.append(train_rms_std_50)
    test_rms_std_50_list.append(test_rms_std_50)

    # 75%
    train_rms_std_75, test_rms_std_75, _ = fit_ml(train_feature, train_target, test_feature,
                                                  test_target, train_mean, train_std * 0.75,
                                                  phi_sigmoid)

    train_rms_std_75_list.append(train_rms_std_75)
    test_rms_std_75_list.append(test_rms_std_75)

    # 100%
    train_rms_std_100, test_rms_std_100, _ = fit_ml(train_feature, train_target, test_feature,
                                                    test_target, train_mean, train_std, phi_sigmoid)

    train_rms_std_100_list.append(train_rms_std_100)
    test_rms_std_100_list.append(test_rms_std_100)

    # 125%
    train_rms_std_125, test_rms_std_125, _ = fit_ml(train_feature, train_target, test_feature,
                                                    test_target, train_mean, train_std * 1.25,
                                                    phi_sigmoid)

    train_rms_std_125_list.append(train_rms_std_125)
    test_rms_std_125_list.append(test_rms_std_125)

    # 150%
    train_rms_std_150, test_rms_std_150, _ = fit_ml(train_feature, train_target, test_feature,
                                                    test_target, train_mean, train_std * 1.5,
                                                    phi_sigmoid)

    train_rms_std_150_list.append(train_rms_std_150)
    test_rms_std_150_list.append(test_rms_std_150)

# %%
print('N-fold maximum likelihood:')

pd.DataFrame(
    data={
        'train rms': [
            np.mean(train_rms_std_50_list),
            np.mean(train_rms_std_75_list),
            np.mean(train_rms_std_100_list),
            np.mean(train_rms_std_125_list),
            np.mean(train_rms_std_150_list),
        ],
        'test rms': [
            np.mean(test_rms_std_50_list),
            np.mean(test_rms_std_75_list),
            np.mean(test_rms_std_100_list),
            np.mean(test_rms_std_125_list),
            np.mean(test_rms_std_150_list),
        ]
    },
    index=[
        '50% std',
        '75% std',
        '100% std',
        '125% std',
        '150% std',
    ],
)

# %% [markdown]
# The result shows that the best std is 100% of the original std.

# %% [markdown]
# ## 3. Maximum A Posteriori Approach
#

# %% [markdown]
# ### (a)

# %% [markdown]
# ### Maximum Likelihood Approach
#
# Maximize the likelihood of the data given the model, where the likelihood $ p(t | \textbf{x}, \textbf{w}, \beta) = \mathcal{N}(t | y(\textbf{x}, \textbf{w}),\beta^{-1}) $.
#
# This approach is equivalent to minimizing the mean square error $ \textit{E}_{D}(\textbf{w}) = \frac{1}{2} \sum^{N}_{n=1}{\{t_{n} - \textbf{w}^{T} \phi(\textbf{x}_{n}) \}} $. The optimal solution is given by the normal equation $ \textbf{w}_{ML} = (\Phi^{T} \Phi)^{-1} \Phi^{T} \textbf{t} $.
#
# ### Maximum A Posteriori Approach
#
# Assume that the prior distribution of the weight is a Gaussian distribution with mean $ 0 $ and variance $ \alpha^{-1} $, which is $ p(\textbf{w} | \alpha) = \mathcal{N}(\textbf{w} | \textbf{0}, \alpha^{-1}\textbf{I}) $.
#
# Maximize the posterior probability of the model given the data, where the posterior probability $ p(\textbf{w} | \textbf{t}, \textbf{x}, \alpha, \beta) = \mathcal{N}(\textbf{w} | \beta(\alpha\textbf{I} + \beta\Phi^{T}\Phi)^{-1}\Phi^{T}\textbf{t}, (\alpha\textbf{I} + \beta\Phi^{T}\Phi)^{-1}) $.
#
# This approach is equivalent to minimizing the mean square error $ \textit{E}_{D}(\textbf{w}) = \frac{1}{2} \sum^{N}_{n=1}{\{t_{n} - \textbf{w}^{T} \phi(\textbf{x}_{n}) \}} + \frac{\lambda}{2} \textbf{w}^{T} \textbf{w} $, where $ \lambda = \frac{\alpha}{\beta} $. The optimal solution is given by the normal equation $ \textbf{w}_{MAP} = (\lambda\textbf{I} + \Phi^{T} \Phi)^{-1} \Phi^{T} \textbf{t} $.
#
# With assuming the distribution of the weight, the model is more robust to the noise of the data. The model with the maximum a posteriori approach is more stable than the model with the maximum likelihood approach.
#


# %%
def optimal_weight_map(phi, target, lamb):
    return np.linalg.inv((np.eye(phi.shape[1]) * lamb + phi.T @ phi)) @ phi.T @ target


def fit_map(train_feature, train_target, test_feature, test_target, mean, std, phi_fn, lamb):
    """Fits the phi function.
    
    Args:
        train_feature: np.ndarray with shape (n_samples, n_features)
        train_target: np.ndarray with shape (n_samples, 1)
        test_feature: np.ndarray with shape (n_samples, n_features)
        test_target: np.ndarray with shape (n_samples, 1)
        mean: np.ndarray with shape (1, n_features)
        std: np.ndarray with shape (1, n_features)
        phi_fn: function with signature (features, mean, std) -> phi
        lamb: float
        
    Returns:
        train_rms: float
        test_rms: float
        weights: np.ndarray with shape (phi.shape[1], 1)
    """
    train_phi = phi_fn(train_feature, mean, std)
    test_phi = phi_fn(test_feature, mean, std)
    weights = optimal_weight_map(train_phi, train_target, lamb)
    train_rms = root_mean_square_error(train_phi @ weights, train_target)
    test_rms = root_mean_square_error(test_phi @ weights, test_target)
    return train_rms, test_rms, weights


# %% [markdown]
# ### (b)

# %% [markdown]
# Compare the sigmoid basis with the maximum likelihood approach and the maximum a posteriori approach.

# %%
LAMBDA = 0.1

train_rms_sigmoid_list, test_rms_sigmoid_list = [], []

for i, (train_index, test_index) in enumerate(n_fold(dataset, 5)):
    train_feature, train_target = dataset[train_index, :-1], dataset[train_index, -1]
    test_feature, test_target = dataset[test_index, :-1], dataset[test_index, -1]

    train_mean, train_std = get_mean_and_std(train_feature)

    train_rms_sigmoid, test_rms_sigmoid, _ = fit_map(train_feature, train_target, test_feature,
                                                     test_target, train_mean, train_std,
                                                     phi_sigmoid, LAMBDA)

    train_rms_sigmoid_list.append(train_rms_sigmoid)
    test_rms_sigmoid_list.append(test_rms_sigmoid)

# %%
print('N-fold maximum likelihood v.s. N-fold maximum a posteriori:')

pd.DataFrame(
    data={
        'train rms': [
            np.mean(train_rms_std_100_list),
            np.mean(train_rms_sigmoid_list),
        ],
        'test rms': [
            np.mean(test_rms_std_100_list),
            np.mean(test_rms_sigmoid_list),
        ]
    },
    index=[
        'sigmoid ml',
        'sigmoid map',
    ],
)

# %% [markdown]
# Maximum a posteriori approach performs better than the maximum likelihood approach in the testing dataset. The model with the maximum a posteriori approach is more general than the model with the maximum likelihood approach, which is consistent with the result in (a).
