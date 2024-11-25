import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from scipy.special import psi
from scipy.stats import pearsonr
from scipy.stats import skew, kurtosis
from collections import Counter, defaultdict
import hsic


# BINARY      = "Binary"
# CATEGORICAL = "Categorical"
# NUMERICAL   = "Numerical"

    
def weighted_mean(values, weights=None):
    if weights == None:
        weights = np.array([1 for _ in range(len(values))])
    average = np.average(values, weights=weights, axis=0)
    return average


def weighted_std(values, weights=None):
    if weights == None:
        weights = np.array([1 for _ in range(len(values))])
    average = np.average(values, weights=weights, axis=0)
    variance = np.dot(weights, (values-average)**2)/weights.sum()  
    return np.sqrt(variance)


def count_unique(x):
    return len(np.unique(x))


def count_unique_ratio(x):
    return len(set(x))/float(len(x))


# def binary(tp):
#     assert type(tp) is str
#     return tp == BINARY


# def categorical(tp):
#     assert type(tp) is str
#     return tp == CATEGORICAL


# def numerical(tp):
#     assert type(tp) is str
#     return tp == NUMERICAL


def binary_entropy(p, base):
    assert p <= 1 and p >= 0
    h = -(p*np.log(p) + (1-p)*np.log(1-p)) if (p != 0) and (p != 1) else 0
    return h/np.log(base)


def discrete_probability(x, ffactor=3, maxdev=3):    
    x = discretized_sequence(x, ffactor, maxdev)
    if isinstance(x, np.ndarray):
        x = x.flatten().tolist()
    return Counter(x)


def discretized_values(x, ffactor=3, maxdev=3): # return 값 list
    if count_unique(x) > (2*ffactor*maxdev+1):
        vmax =  ffactor*maxdev
        vmin = -ffactor*maxdev
        return range(vmin, vmax+1)
    else:
        return sorted(np.unique(x))


def len_discretized_values(x, ffactor, maxdev):
    return len(discretized_values(x, ffactor, maxdev))


def discretized_sequence(x, ffactor, maxdev, norm=True):
    if not norm or (count_unique(x) > len_discretized_values(x, ffactor, maxdev)):
        if norm:
            if np.mean(x) == np.nan:
                x = x/np.std(x)
            else:
                x = (x - np.mean(x))/np.std(x)
            xf = x[abs(x) < maxdev]
            if np.mean(xf) == np.nan:
                x = x/np.std(xf)
            else:
                x = (x - np.mean(xf))/np.std(xf)
        x = np.round(x*ffactor)
        vmax =  ffactor*maxdev
        vmin = -ffactor*maxdev
        x[x > vmax] = vmax
        x[x < vmin] = vmin
    return x


def discretized_sequences(x, y, ffactor=3, maxdev=3):
    return discretized_sequence(x, ffactor, maxdev), discretized_sequence(y, ffactor, maxdev)


def normalized_error_probability(x, y, ffactor=3, maxdev=3):
    x, y = discretized_sequences(x, y, ffactor, maxdev)
    if isinstance(x, np.ndarray):
        x = x.flatten().tolist()
    if isinstance(y, np.ndarray):
        y = y.flatten().tolist()
    cx = Counter(x)
    cy = Counter(y)
    nx = len(cx)
    ny = len(cy)
    pxy = defaultdict(lambda: 0)
    for p in zip(x, y):
        pxy[p] += 1
    pxy = np.array([[pxy[(a,b)] for b in cy] for a in cx], dtype = float)
    pxy = pxy/pxy.sum()
    perr = 1 - np.sum(pxy.max(axis=1))
    max_perr = 1 - np.max(pxy.sum(axis=0))
    pnorm = perr/max_perr if max_perr > 0 else perr
    return pnorm


def discrete_entropy(x, ffactor=3, maxdev=3, bias_factor=0.7):
    c = discrete_probability(x, ffactor, maxdev)
    pk = np.array(list(c.values()), dtype=float)
    pk = pk/pk.sum()
    vec = pk*np.log(pk)
    S = -np.sum(vec, axis=0)
    return S + bias_factor*(len(pk) - 1)/float(2*len(x))


def discrete_divergence(cx, cy): #
    for a, v in cx.most_common():
        if cy[a] == 0: cy[a] = 1

    nx = float(sum(cx.values()))
    ny = float(sum(cy.values()))
    sum = 0.
    for a, v in cx.most_common():
        px = v/nx
        py = cy[a]/ny
        sum += px*np.log(px/py)
    return sum


def discrete_joint_entropy(x, y, ffactor=3, maxdev=3):
    x, y = discretized_sequences(x, y, ffactor, maxdev)
    return discrete_entropy(np.stack((x, y), axis=-1))


def normalized_discrete_joint_entropy(x, y, ffactor=3, maxdev=3):
    x, y = discretized_sequences(x, y, ffactor, maxdev)
    e = discrete_entropy(np.stack((x, y), axis=-1))
    nx = len_discretized_values(x, ffactor, maxdev)
    ny = len_discretized_values(y, ffactor, maxdev)
    if nx*ny>0: 
        e = e/np.log(nx*ny)
    if e == np.nan:
        e = 0
    return e


def discrete_conditional_entropy(x, y):
    return discrete_joint_entropy(x, y) - discrete_entropy(y)


def adjusted_mutual_information(x, y, ffactor=3, maxdev=3):
    x, y = discretized_sequences(x, y, ffactor, maxdev)
    return adjusted_mutual_info_score(x, y)


def discrete_mutual_information(x, y):
    ex = discrete_entropy(x)
    ey = discrete_entropy(y)
    exy = discrete_joint_entropy(x, y)
    mxy = max((ex + ey) - exy, 0) # Mutual information is always positive: max() avoid negative values due to numerical errors
    return mxy


def normalized_discrete_entropy(x, ffactor=3, maxdev=3):
    e = discrete_entropy(x, ffactor, maxdev)
    n = len_discretized_values(x, ffactor, maxdev)
    if n>0: 
        e = e/np.log(n)
    if e == np.nan:
        e = 0
    return e


# Continuous information measures
def to_numerical(x, y):
    dx = defaultdict(lambda: np.zeros(2))
    for i, a in enumerate(x):
        dx[a][0] += y[i]
        dx[a][1] += 1
    for a in dx.keys():
        dx[a][0] /= dx[a][1]
    x = np.array([dx[a][0] for a in x], dtype=float)
    return x


def normalize(x):
    if np.mean(x) != np.nan:
        x = x - np.mean(x)
    if np.std(x) > 0:
        x = x/np.std(x)
    return x


def normalized_entropy_baseline(x):
    if len(set(x)) < 2:
        return 0
    x = normalize(x)
    xs = np.sort(x)
    delta = xs[1:] - xs[:-1]
    delta = delta[delta != 0]
    hx = np.mean(np.log(delta))
    hx += psi(len(delta))
    hx -= psi(1)
    return hx
    

def normalized_entropy(x, m=2):
    x = normalize(x)
    cx = Counter(x)
    if len(cx) < 2:
        return 0
    xk = np.array(list(cx.keys()), dtype=float)
    xk.sort()
    delta = (xk[1:] - xk[:-1])/m
    counter = np.array([cx[i] for i in xk], dtype=float)
    hx = np.sum(counter[1:]*np.log(delta/counter[1:]))/len(x)
    hx += (psi(len(delta)) - np.log(len(delta)))
    hx += np.log(len(x))
    hx -= (psi(m) - np.log(m))
    return hx


def igci(x, y):
    if len(set(x)) < 2:
        return 0
    x = normalize(x)
    y = normalize(y)
    if len(x) != len(set(x)):
        dx = defaultdict(lambda: np.zeros(2))
        for i, a in enumerate(x):
            dx[a][0] += y[i]
            dx[a][1] += 1
        for a in dx.keys():
            dx[a][0] /= dx[a][1]
        xy = np.array(sorted([[a, dx[a][0]] for a in dx.keys()]), dtype=float)
        counter = np.array([dx[a][1] for a in xy[:,0]], dtype=float)
    else:
        xy = np.array(sorted(zip(x, y)), dtype = float)
        counter = np.ones(len(x))
    delta = xy[1:] - xy[:-1]
    selec = delta[:,1] != 0
    delta = delta[selec]
    counter = np.min([counter[1:], counter[:-1]], axis=0)
    counter = counter[selec]
    hxy = np.sum(counter*np.log(delta[:,0]/np.abs(delta[:,1])))/len(x)
    return hxy


def uniform_divergence(x, m=2):
    x = normalize(x)
    cx = Counter(x)
    xk = np.array(list(cx.keys()), dtype=float)
    xk.sort()
    delta = np.zeros(len(xk))
    if len(xk) > 1:
        delta[0] = xk[1]-xk[0]
        delta[1:-1] = (xk[m:]-xk[:-m])/m
        delta[-1] = xk[-1]-xk[-2]
    else:
        delta = np.array(np.sqrt(12))
    counter = np.array([cx[i] for i in xk], dtype=float)
    delta = delta/np.sum(delta)
    hx = np.sum(counter*np.log(counter/delta))/len(x)
    hx -= np.log(len(x))
    hx += (psi(m) - np.log(m))
    return hx


def normalized_skewness(x):
    y = skew(normalize(x))
    if np.isnan(y):
        return 0
    else:
        return y


def normalized_kurtosis(x):
    y = kurtosis(normalize(x))
    if np.isnan(y):
        return 0
    else:
        return y


def normalized_moment(x, y, n, m):
    x = normalize(x)
    y = normalize(y)
    return np.mean((x**n)*(y**m))


def moment21(x, y):
    return normalized_moment(x, y, 2, 1)


def moment22(x, y):
    return normalized_moment(x, y, 2, 2)


def moment31(x, y):
    return normalized_moment(x, y, 3, 1)


def fit(x, y):
    if (count_unique(x) <= 2) or (count_unique(y) <= 2):
        return 0
    x = normalize(x)
    y = normalize(y)
    xy1 = np.polyfit(x, y, 1)
    xy2 = np.polyfit(x, y, 2)
    return abs(2*xy2[0]) + abs(xy2[1]-xy1[0])


def fit_error(x, y, m=2):
    x = normalize(x)
    y = normalize(y)
    if (count_unique(x) <= m) or (count_unique(y) <= m):
        xy = np.polyfit(x, y, min(count_unique(x), count_unique(y))-1)
    else:
        xy = np.polyfit(x, y, m)
    return np.std(y - np.polyval(xy, x))


def fit_noise_entropy(x, y, ffactor=3, maxdev=3, minc=10):
    x, y = discretized_sequences(x, y, ffactor, maxdev)
    cx = Counter(x)
    entyx = []
    for a in cx.keys():
        if cx[a] > minc:
            entyx.append(discrete_entropy(y[x==a]))
    if len(entyx) == 0: return 0
    n = len_discretized_values(y, ffactor, maxdev)
    return np.std(entyx)/np.log(n)


def fit_noise_skewness(x, y, ffactor=3, maxdev=3, minc=8):
    xd, yd = discretized_sequences(x, y, ffactor, maxdev)
    cx = Counter(xd)
    skewyx = []
    for a in cx.keys():
        if cx[a] >= minc:
            skewyx.append(normalized_skewness(y[xd==a]))
    if len(skewyx) == 0: return 0
    return np.std(skewyx)


def fit_noise_kurtosis(x, y, ffactor=3, maxdev=3, minc=8):
    xd, yd = discretized_sequences(x, y, ffactor, maxdev)
    cx = Counter(xd)
    kurtyx = []
    for a in cx.keys():
        if cx[a] >= minc:
            kurtyx.append(normalized_kurtosis(y[xd==a]))
    if len(kurtyx) == 0: return 0
    return np.std(kurtyx)


def conditional_distribution_similarity(x, y, ffactor=2, maxdev=3, minc=12):
    xd, yd = discretized_sequences(x, y, ffactor, maxdev)
    cx = Counter(xd)
    cy = Counter(yd)
    yrange = sorted(cy.keys())
    ny = len(yrange)
    py = np.array([cy[i] for i in yrange], dtype=float)
    py = py/py.sum()
    pyx = []
    for a in cx.keys():
        if cx[a] > minc:
            yx = y[xd==a]
            # if not numerical(ty):
            #     cyx = Counter(yx)
            #     pyxa = np.array([cyx[i] for i in yrange], dtype=float)
            #     pyxa.sort()
            if count_unique(y) > len_discretized_values(y, ffactor, maxdev):
                if np.mean(yx) == np.nan:
                    yx = yx/np.std(y)
                else:
                    yx = (yx - np.mean(yx))/np.std(y)
                yx = discretized_sequence(yx, ffactor, maxdev, norm=False)
                cyx = Counter(yx.astype(int))
                pyxa = np.array([cyx[i] for i in discretized_values(y, ffactor, maxdev)], dtype=float)
            else:
                cyx = Counter(yx)
                pyxa = [cyx[i] for i in yrange]
                pyxax = np.array([0]*(ny-1) + pyxa + [0]*(ny-1), dtype=float)
                xcorr = [sum(py*pyxax[i:i+ny]) for i in range(2*ny-1)]
                imax = xcorr.index(max(xcorr))
                pyxa = np.array([0]*(2*ny-2-imax) + pyxa + [0]*imax, dtype=float)
            assert pyxa.sum() == cx[a]
            pyxa = pyxa/pyxa.sum()
            pyx.append(pyxa)
    
    if len(pyx)==0: return 0           
    pyx = np.array(pyx)
    pyx = pyx - pyx.mean(axis=0)
    return np.std(pyx)


def correlation(x, y):
    x = normalize(x)
    y = normalize(y)
    r = pearsonr(x, y)[0]
    if r == np.nan:
        r = 0
    return r


def normalized_hsic(x, y):
    x = normalize(x)
    y = normalize(y)
    h = hsic.FastHsicTestGamma(x, y)
    return h

