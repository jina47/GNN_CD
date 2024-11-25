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


# edge_features = [
#     ('Sub_Numerical', PairColumnTransform(operator.sub), None),
#     ('Sub_Normalized Entropy Baseline', PairColumnTransform(operator.sub), SimpleTransform(ft.normalized_entropy_baseline)),
#     ('Sub_Normalized Entropy', PairColumnTransform(operator.sub), SimpleTransform(ft.normalized_entropy)),
#     ('IGCI', PairColumnTransform(ft.igci), None),
#     ('Sub_Uniform Divergence', PairColumnTransform(operator.sub), SimpleTransform(ft.uniform_divergence)),
#     ('Polyfit', PairColumnTransform(ft.fit), None),
#     ('Polyfit Error', PairColumnTransform(ft.fit_error), None),
#     # ('Normalized Error Probability', PairColumnTransform(ft.normalized_error_probability), None),
#     ('Conditional Distribution Entropy Variance', PairColumnTransform(ft.fit_noise_entropy), None),
#     ('Conditional Distribution Skewness Variance', PairColumnTransform(ft.fit_noise_skewness), None),
#     ('Conditional Distribution Kurtosis Variance', PairColumnTransform(ft.fit_noise_kurtosis), None),
#     ('Conditional Distribution Similarity', PairColumnTransform(ft.conditional_distribution_similarity), None),
#     ('Moment21', PairColumnTransform(ft.moment21), None),
#     ('Moment22', PairColumnTransform(ft.moment22), None),
#     ('Moment31', PairColumnTransform(ft.moment31), None),
#     ('Sub_Skewness', PairColumnTransform(operator.sub), SimpleTransform(ft.normalized_skewness)),
#     ('Sub_Skewness', PairColumnTransform(operator.sub), SimpleTransform(ft.normalized_kurtosis)),

# ]

# all_features = [
#     ('Numerical', SimpleTransform(ft.numerical)),
#     ('Sub_Numerical', MultiColumnTransform(operator.sub)),
#     ('Abs_Sub_Numerical', SimpleTransform(abs)), # edge
    
#     ('Normalized Entropy Baseline', MultiColumnTransform(ft.normalized_entropy_baseline)),
#     ('Max_Normalized Entropy Baseline', MultiColumnTransform(max)), # edge
#     ('Min_Normalized Entropy Baseline', MultiColumnTransform(min)), # edge
#     ('Sub_Normalized Entropy Baseline', MultiColumnTransform(operator.sub)), # edge
#     ('Abs_Sub_Normalized Entropy Baseline', SimpleTransform(abs)), # edge
    
#     ('Normalized Entropy', MultiColumnTransform(ft.normalized_entropy)),
#     ('Max_Normalized Entropy', MultiColumnTransform(max)), # edge
#     ('Min_Normalized Entropy', MultiColumnTransform(min)), # edge
#     ('Sub_Normalized Entropy', MultiColumnTransform(operator.sub)), # edge
#     ('Abs_Sub_Normalized Entropy', SimpleTransform(abs)), # edge
    
#     ('IGCI', MultiColumnTransform(ft.igci)),
#     ('Sub_IGCI', MultiColumnTransform(operator.sub)), # edge
#     ('Abs_IGCI', SimpleTransform(abs)), # edge
    
#     ('Uniform Divergence', MultiColumnTransform(ft.uniform_divergence)),
#     ('Max_Uniform Divergence', MultiColumnTransform(max)), # edge
#     ('Min_Uniform Divergence', MultiColumnTransform(min)), # edge
#     ('Sub_Uniform Divergence', MultiColumnTransform(operator.sub)), # edge
#     ('Abs_Sub_Uniform Divergence', SimpleTransform(abs)), # edge
    
#     ('Discrete Entropy', MultiColumnTransform(ft.discrete_entropy)),
#     ('Max_Discrete Entropy', MultiColumnTransform(max)), # edge
#     ('Min_Discrete Entropy', MultiColumnTransform(min)), # edge
#     ('Sub_Discrete Entropy', MultiColumnTransform(operator.sub)), # edge
#     ('Abs_Sub_Discrete Entropy', SimpleTransform(abs)), # edge
    
#     ('Normalized Discrete Entropy', MultiColumnTransform(ft.normalized_discrete_entropy)),
#     ('Max_Normalized Discrete Entropy', MultiColumnTransform(max)),
#     ('Min_Normalized Discrete Entropy', MultiColumnTransform(min)),
#     ('Sub_Normalized Discrete Entropy', MultiColumnTransform(operator.sub)),
#     ('Abs_Sub_Normalized Discrete Entropy', SimpleTransform(abs)),
    
#     ('Discrete Joint Entropy', MultiColumnTransform(ft.discrete_joint_entropy)),
#     ('Normalized Discrete Joint Entropy', MultiColumnTransform(ft.normalized_discrete_joint_entropy)),
#     ('Discrete Conditional Entropy', MultiColumnTransform(ft.discrete_conditional_entropy)),
#     ('Discrete Mutual Information', MultiColumnTransform(ft.discrete_mutual_information)),
#     ('Normalized Discrete Mutual Information', ['Discrete Mutual Information[A,A type,B,B type]','Min[Discrete Entropy[A,A type],Discrete Entropy[B,B type]]'], MultiColumnTransform(operator.truediv)),
#     ('Normalized Discrete Mutual Information', ['Discrete Mutual Information[A,A type,B,B type]','Discrete Joint Entropy[A,A type,B,B type]'], MultiColumnTransform(operator.truediv)),
#     ('Adjusted Mutual Information', ['A','A type','B','B type'], MultiColumnTransform(ft.adjusted_mutual_information)),

#     ('Polyfit', MultiColumnTransform(ft.fit)),
#     ('Sub_Polyfit', MultiColumnTransform(operator.sub)),
#     ('Abs_Sub_Polyfit', SimpleTransform(abs)),

#     ('Polyfit Error', MultiColumnTransform(ft.fit_error)),
#     ('Sub_Polyfit Error', MultiColumnTransform(operator.sub)),
#     ('Abs_Sub_Polyfit Error', SimpleTransform(abs)),

#     ('Normalized Error Probability', MultiColumnTransform(ft.normalized_error_probability)),
#     ('Sub_Normalized Error Probability', MultiColumnTransform(operator.sub)),
#     ('Abs_Sub_Normalized Error Probability', SimpleTransform(abs)),

#     ('Conditional Distribution Entropy Variance', MultiColumnTransform(ft.fit_noise_entropy)),
#     ('Sub_Conditional Distribution Entropy Variance', MultiColumnTransform(operator.sub)),
#     ('Abs_Sub_Conditional Distribution Entropy Variance', SimpleTransform(abs)),

#     ('Conditional Distribution Skewness Variance', MultiColumnTransform(ft.fit_noise_skewness)),
#     ('Sub_Conditional Distribution Skewness Variance', MultiColumnTransform(operator.sub)),
#     ('Abs_Sub_Conditional Distribution Skewness Variance', SimpleTransform(abs)),

#     ('Conditional Distribution Kurtosis Variance', MultiColumnTransform(ft.fit_noise_kurtosis)),
#     ('Sub_Conditional Distribution Kurtosis Variance', MultiColumnTransform(operator.sub)),
#     ('Abs_Conditional Distribution Kurtosis Variance', SimpleTransform(abs)),

#     ('Conditional Distribution Similarity', MultiColumnTransform(ft.conditional_distribution_similarity)),
#     ('Sub_Conditional Distribution Similarity', MultiColumnTransform(operator.sub)),
#     ('Abs', SimpleTransform(abs)),

#     ('Moment21', MultiColumnTransform(ft.moment21)),
#     ('Sub_Moment21', MultiColumnTransform(operator.sub)),
#     ('Abs_Sub_Moment21', SimpleTransform(abs)),
    
#     ('Abs', 'Moment21[A,A type,B,B type]', SimpleTransform(abs)),
#     ('Abs', 'Moment21[B,B type,A,A type]', SimpleTransform(abs)),
#     ('Sub', ['Abs[Moment21[A,A type,B,B type]]','Abs[Moment21[B,B type,A,A type]]'], MultiColumnTransform(operator.sub)),
#     ('Abs', 'Sub[Abs[Moment21[A,A type,B,B type]],Abs[Moment21[B,B type,A,A type]]]', SimpleTransform(abs)),
    
#     ('Moment31', MultiColumnTransform(ft.moment31)),
#     ('Sub_Moment31', MultiColumnTransform(operator.sub)),
#     ('Abs_sub_Moment31',SimpleTransform(abs)),

#     ('Abs','Moment31[A,A type,B,B type]', SimpleTransform(abs)),
#     ('Sub', ['Abs[Moment31[A,A type,B,B type]]','Abs[Moment31[B,B type,A,A type]]'], MultiColumnTransform(operator.sub)),
#     ('Abs','Sub[Abs[Moment31[A,A type,B,B type]],Abs[Moment31[B,B type,A,A type]]]', SimpleTransform(abs)),

#     ('Skewness', MultiColumnTransform(ft.normalized_skewness)),
#     ('Sub_Skewness', MultiColumnTransform(operator.sub)),
#     ('Abs_Sub_Skewness', SimpleTransform(abs)),
    
#     ('Abs', 'Skewness[A,A type]', SimpleTransform(abs)),
#     ('Max', ['Abs[Skewness[A,A type]]','Abs[Skewness[B,B type]]'], MultiColumnTransform(max)),
#     ('Min', ['Abs[Skewness[A,A type]]','Abs[Skewness[B,B type]]'], MultiColumnTransform(min)),
#     ('Sub', ['Abs[Skewness[A,A type]]','Abs[Skewness[B,B type]]'], MultiColumnTransform(operator.sub)),
#     ('Abs', 'Sub[Abs[Skewness[A,A type]],Abs[Skewness[B,B type]]]', SimpleTransform(abs)),
    
#     ('Kurtosis', MultiColumnTransform(ft.normalized_kurtosis)),
#     ('Max_Kurtosis', MultiColumnTransform(max)),
#     ('Min_Kurtosis', MultiColumnTransform(min)),
#     ('Sub_Kurtosis', MultiColumnTransform(operator.sub)),
#     ('Abs_Sub_Kurtosis', SimpleTransform(abs)),

#     ('HSIC', MultiColumnTransform(ft.normalized_hsic)),
#     ('Pearson R', MultiColumnTransform(ft.correlation)),
#     ('Abs_Pearson R', SimpleTransform(abs))
#     ]




def calculate_method(args):
    obj = args[0] # Transform
    name = args[1] # function - fit_transform
    margs = args[2] # (X[column_names], y)
    method = getattr(obj, name)
    return method(*margs)


def extract_node_features(X, features=node_features, y=None, n_jobs=-1):
    if n_jobs != 1:
        pool = Pool(n_jobs if n_jobs != -1 else None)
        pmap = pool.map
    else:
        pmap = map
        
    # def complete_feature_name(feature_name, column_names):
    #     if type(column_names) is list:
    #         long_feature_name = feature_name + '[' + ','.join(column_names) + ']'
    #     else:
    #         long_feature_name = feature_name + '[' + column_names + ']'
    #     if feature_name[0] == '+':
    #         long_feature_name = long_feature_name[1:]
    #     return long_feature_name
    
    # def is_in_X(column_names):
    #     if type(column_names) is list:
    #         return set(column_names).issubset(X.columns)
    #     else:
    #         return column_names in X.columns
        
    # def can_be_extracted(feature_name, column_names):
    #     long_feature_name = complete_feature_name(feature_name, column_names)
    #     to_be_extracted = ((feature_name[0] == '+') or (long_feature_name not in X.columns))
    #     return to_be_extracted and is_in_X(column_names)

    # while True:
    #     new_features_list = [(complete_feature_name(feature_name, column_names), column_names, extractor) 
    #         for feature_name, column_names, extractor in features if can_be_extracted(feature_name, column_names)]
    #     if not new_features_list:
    #         break
    #     task = [(extractor, 'fit_transform', (X[column_names], y)) for _, column_names, extractor in new_features_list]
    #     new_features = pmap(calculate_method, task)
    #     for (feature_name, _, _), feature in zip(new_features_list, new_features):
    #         X[feature_name] = feature
    
    result = []

    for name, extractor in features:
        y = extractor.transform(X)
        if len(result) == 0:
            result = y
        else:
            result = np.column_stack([result, y])

    # node_data = pd.DataFrame(result) # dataframe으로 고칠지 말지는 나중에 고민 바로 GNN 넣을거면 안 바꿔도 될 듯? 근데 나중에 column 명은 필요할텐데....
    node_data = result
    return node_data


def extract_edge_features(X, features=edge_features, y=None, n_jobs=-1):
    if n_jobs != 1:
        pool = Pool(n_jobs if n_jobs != -1 else None)
        pmap = pool.map
    else:
        pmap = map
        
    # def complete_feature_name(feature_name, column_names):
    #     if type(column_names) is list:
    #         long_feature_name = feature_name + '[' + ','.join(column_names) + ']'
    #     else:
    #         long_feature_name = feature_name + '[' + column_names + ']'
    #     if feature_name[0] == '+':
    #         long_feature_name = long_feature_name[1:]
    #     return long_feature_name
    
    # def is_in_X(column_names):
    #     if type(column_names) is list:
    #         return set(column_names).issubset(X.columns)
    #     else:
    #         return column_names in X.columns
        
    # def can_be_extracted(feature_name, column_names):
    #     long_feature_name = complete_feature_name(feature_name, column_names)
    #     to_be_extracted = ((feature_name[0] == '+') or (long_feature_name not in X.columns))
    #     return to_be_extracted and is_in_X(column_names)

    # while True:
    #     new_features_list = [(complete_feature_name(feature_name, column_names), column_names, extractor) 
    #         for feature_name, column_names, extractor in features if can_be_extracted(feature_name, column_names)]
    #     if not new_features_list:
    #         break
    #     task = [(extractor, 'fit_transform', (X[column_names], y)) for _, column_names, extractor in new_features_list]
    #     new_features = pmap(calculate_method, task)
    #     for (feature_name, _, _), feature in zip(new_features_list, new_features):
    #         X[feature_name] = feature

    nodes = [n for n in range(len(X.columns))]
    pairs = list(itertools.combinations(nodes, 2)) # node의 컬럼인덱스 혹은 컬럼명
    # pairs
    result = []
    for name, extractor, operation in features: # column 이름...... 어떻게 할지 나중에 생각
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
            # print(m)
            # print(n)
            # print()
            y = np.mean(np.abs(m))
            z = np.mean(np.abs(n))
            pair_values1.append(y)
            pair_values2.append(z)
        result.append(pair_values1)
        # if pair_values1 != pair_values2:
        result.append(pair_values2)

    result = np.array(result).T
    # print(result.shape)
    # edge_data = pd.DataFrame(result) # numpy로 할지 말지 다시 보기
    edge_data = result
    return edge_data
