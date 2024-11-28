import features as ft
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import operator
import math
from multiprocessing import Pool
import itertools



class SimpleTransform(BaseEstimator):
    def __init__(self, transformer, *args):
        self.transformer = transformer
        self.args = args

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        if self.args:
            return np.array([self.transformer(X, *self.args)], ndmin=2).T
        # return np.array([self.transformer(x) for x in X], ndmin=2).T
        else:
            return np.array([self.transformer(X)], ndmin=2).T


class MultiColumnTransform(BaseEstimator):
    def __init__(self, transformer, *args):
        self.transformer = transformer
        self.args = args

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        X = X.to_numpy()
        if self.args:
            return np.array([self.transformer(column, *self.args) for column in X.T], ndmin=2).T
        else:
            # return X.apply(self.transformer, axis=0).to_numpy().T
            return np.array([self.transformer(column) for column in X.T], ndmin=2).T


class PairColumnTransform(BaseEstimator):
    def __init__(self, transformer, *args):
        self.transformer = transformer
        self.args = args

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X1, X2, y=None):
        return self.transform(X1, X2)

    def transform(self, X1, X2, y=None):
        if self.args:
            # return np.array([self.transformer(c1, c2, *self.args) for c1, c2 in zip(X1.T, X2.T)], ndmin=2).T
            return np.array(self.transformer(X1.T, X2.T, *self.args), ndmin=2).T
        else:
            # return np.array([self.transformer(c1, c2) for c1, c2 in zip(X1.T, X2.T)], ndmin=2).T
            return np.array(self.transformer(X1.T, X2.T), ndmin=2).T


node_features = [
    ('Max', MultiColumnTransform(np.max)),
    ('Min', MultiColumnTransform(np.min)),
    ('Normalized Entropy Baseline', MultiColumnTransform(ft.normalized_entropy_baseline)),
    ('Normalized Entropy', MultiColumnTransform(ft.normalized_entropy)),
    ('Uniform Divergence', MultiColumnTransform(ft.uniform_divergence)),
    ('Skewness', MultiColumnTransform(ft.normalized_skewness)),
    ('Kurtosis', MultiColumnTransform(ft.normalized_kurtosis)),
    ('weighted_mean', MultiColumnTransform(ft.weighted_mean)),
    ('weighted_std', MultiColumnTransform(ft.weighted_std)),
    ('Interquartile Range', MultiColumnTransform(lambda x: np.percentile(x, 75) - np.percentile(x, 25))),
    # ('Median', MultiColumnTransform(np.median)),
    ('Coefficient of Variation', MultiColumnTransform(lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else 0)),
    # ('Shannon Diversity Index', MultiColumnTransform(lambda x: -np.sum(x * np.log(x)) if np.all(x > 0) else 0)),
]


edge_features = [
    ('Sub_Numerical', PairColumnTransform(operator.sub), None),
    ('Sub_Normalized Entropy Baseline', PairColumnTransform(operator.sub), SimpleTransform(ft.normalized_entropy_baseline)),
    ('Sub_Normalized Entropy', PairColumnTransform(operator.sub), SimpleTransform(ft.normalized_entropy)),
    ('Sub_Normalized Entropy5', PairColumnTransform(operator.sub), SimpleTransform(ft.normalized_entropy, 5)),
    ('Sub_Normalized Entropy5', PairColumnTransform(operator.sub), SimpleTransform(ft.normalized_entropy, 10)),
    ('IGCI', PairColumnTransform(ft.igci), None),
    ('Sub_Uniform Divergence', PairColumnTransform(operator.sub), SimpleTransform(ft.uniform_divergence)),
    # ('Sub_Uniform Divergence5', PairColumnTransform(operator.sub), SimpleTransform(ft.uniform_divergence, 5)),
    ('Polyfit', PairColumnTransform(ft.fit), None),
    ('Polyfit Error', PairColumnTransform(ft.fit_error), None),
    ('Polyfit Error3', PairColumnTransform(ft.fit_error, 3), None),
    ('Polyfit Error5', PairColumnTransform(ft.fit_error, 5), None),
    ('fit_noise_entropy', PairColumnTransform(ft.fit_noise_entropy), None),
    ('fit_noise_entropy55', PairColumnTransform(ft.fit_noise_entropy, 5, 5), None),
    ('fit_noise_entropy43', PairColumnTransform(ft.fit_noise_entropy, 4, 3), None),
    ('fit_noise_skewness', PairColumnTransform(ft.fit_noise_skewness), None),
    ('fit_noise_skewness55', PairColumnTransform(ft.fit_noise_skewness, 5, 5), None),
    ('fit_noise_kurtosis', PairColumnTransform(ft.fit_noise_kurtosis), None),
    ('fit_noise_kurtosis55', PairColumnTransform(ft.fit_noise_kurtosis, 5, 5), None),
    ('Conditional Distribution Similarity', PairColumnTransform(ft.conditional_distribution_similarity), None),
    ('Conditional Distribution Similarity35', PairColumnTransform(ft.conditional_distribution_similarity, 3, 5), None),
    ('Conditional Distribution Similarity53', PairColumnTransform(ft.conditional_distribution_similarity, 5, 3), None),
    ('normalized_moment23', PairColumnTransform(ft.normalized_moment, 2, 3), None),
    ('normalized_moment33', PairColumnTransform(ft.normalized_moment, 3, 3), None),
    ('Moment21', PairColumnTransform(ft.moment21), None),
    ('Moment22', PairColumnTransform(ft.moment22), None),
    ('Moment31', PairColumnTransform(ft.moment31), None),
    ('Sub_Skewness', PairColumnTransform(operator.sub), SimpleTransform(ft.normalized_skewness)),
    ('Sub_Kurtosis', PairColumnTransform(operator.sub), SimpleTransform(ft.normalized_kurtosis)),
    ('Sub_discrete_entropy', PairColumnTransform(operator.sub), SimpleTransform(ft.discrete_entropy)),
    ('Sub_discrete_entropy53', PairColumnTransform(operator.sub), SimpleTransform(ft.discrete_entropy, 5, 3)),
    ('Sub_discrete_entropy34', PairColumnTransform(operator.sub), SimpleTransform(ft.discrete_entropy, 3, 4)),
    ('Sub_mormalized_discrete_entropy', PairColumnTransform(operator.sub), SimpleTransform(ft.normalized_discrete_entropy)),
    ('Sub_mormalized_discrete_entropy53', PairColumnTransform(operator.sub), SimpleTransform(ft.normalized_discrete_entropy, 5, 3)),
    ('Sub_mormalized_discrete_entropy34', PairColumnTransform(operator.sub), SimpleTransform(ft.normalized_discrete_entropy, 3, 4)),
    ('Normalized Error Probability', PairColumnTransform(ft.normalized_error_probability), None),
    ('Normalized Error Probability53', PairColumnTransform(ft.normalized_error_probability, 5, 3), None),
    ('Normalized Error Probability34', PairColumnTransform(ft.normalized_error_probability, 3, 4), None),
    ('Discrete_joint_entropy', PairColumnTransform(ft.discrete_joint_entropy), None),
    ('Discrete_joint_entropy53', PairColumnTransform(ft.discrete_joint_entropy, 5, 3), None),
    ('Discrete_joint_entropy34', PairColumnTransform(ft.discrete_joint_entropy, 3, 4), None),
    ('Normalized Discrete joint entropy', PairColumnTransform(ft.normalized_discrete_joint_entropy), None),
    ('Normalized Discrete joint entropy53', PairColumnTransform(ft.normalized_discrete_joint_entropy, 5, 3), None),
    ('Normalized Discrete joint entropy34', PairColumnTransform(ft.normalized_discrete_joint_entropy, 3, 4), None),
    ('Discrete conditional entropy', PairColumnTransform(ft.discrete_conditional_entropy), None),
    ('adjusted_mutual_information', PairColumnTransform(ft.adjusted_mutual_information), None),
    ('adjusted_mutual_information53', PairColumnTransform(ft.adjusted_mutual_information, 5, 3), None),
    ('adjusted_mutual_information34', PairColumnTransform(ft.adjusted_mutual_information, 3, 4), None),
    ('discrete_mutual_information', PairColumnTransform(ft.discrete_mutual_information), None),
    ('correlation', PairColumnTransform(ft.correlation), None),
    ('normalized_hsic', PairColumnTransform(ft.normalized_hsic), None),

]



def calculate_method(args):
    obj = args[0] 
    name = args[1] 
    margs = args[2] 
    method = getattr(obj, name)
    return method(*margs)


def extract_node_features(X, features=node_features, y=None, n_jobs=-1):
    if n_jobs != 1:
        pool = Pool(n_jobs if n_jobs != -1 else None)
        pmap = pool.map
    else:
        pmap = map

    result = []

    for name, extractor in features:
        y = extractor.transform(X)
        if len(result) == 0:
            result = y
        else:
            result = np.column_stack([result, y])

    return result


def extract_edge_features(X, features=edge_features, y=None, n_jobs=-1):
    if n_jobs != 1:
        pool = Pool(n_jobs if n_jobs != -1 else None)
        pmap = pool.map
    else:
        pmap = map

    nodes = [n for n in range(len(X.columns))]
    pairs = list(itertools.combinations(nodes, 2)) 
    # pairs
    result = []
    for name, extractor, operation in features: 
        pair_values1 = []
        pair_values2 = []
        for n1, n2 in pairs:
            df1 = X.iloc[:, n1].to_numpy()
            df2 = X.iloc[:, n2].to_numpy()
            if operation:
                df1 = operation.transform(df1)
                df2 = operation.transform(df2)
            m = extractor.transform(df1, df2) 
            n = extractor.transform(df2, df1) 

            y = np.mean(np.abs(m))
            z = np.mean(np.abs(n))
            pair_values1.append(y)
            pair_values2.append(z)
        result.append(pair_values1)
        result.append(pair_values2)

    result = np.array(result).T

    return result
