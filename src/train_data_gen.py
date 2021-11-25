import numpy as np
import pandas as pd
from proxy_label_assignment import get_proxy_label
from feature_utils import get_compatibility_feature, compatibility_score_naive
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

np.random.seed(123)

def check_gender_and_orientation(X, Y):
    """Returns whether Y is sexually compatible with X."""
    def isMale(X):
        return True if X['sex']=='m' else False
    if X['orientation'] == 'straight':
        return isMale(X) != isMale(Y)
    elif X['orientation'] == 'gay':
        return isMale(X) == isMale(Y)
    elif X['orientation'] == 'bisexual':
        return True


def check_location_compatibility(X, Y):
    """Returns whether X and Y are compatible location-wise."""
    return X['location'] == Y['location']


def iscandidate(X, Y):
    """Returns whether X and Y are mutually valid candidates.
    Reciprocal compatibility is checked by our framework."""
    return check_gender_and_orientation(X, Y) and \
        check_gender_and_orientation(Y, X) and \
            check_location_compatibility(X, Y)


def generate_traindata_for_profile(i, df, num_pairs):
    """Generates training pairs for A

    Args:
        i ([type]): row index of profile under consideration
        df ([type]): Complete dataset of profiles available for training
        num_pairs (int, optional): max number of candidate pairs to be sampled

    Returns:
        [list[list[float]], list[int]]: returns X, y proxy training 
        data for a particular profile.
    """
    A = df.iloc[i].to_dict()
    matches = df[df.apply(lambda row: iscandidate(A, row.to_dict()), axis=1)]
    matches = matches.sample(min(len(matches), num_pairs))
    
    X, y = [], []
    for i, row in matches.iterrows():
        X.append(get_compatibility_feature(A, row.to_dict()))
        y.append(get_proxy_label(A, row.to_dict()))

    return X, y

def generate_trainset(df, per_profile_pairs=10, save=True):
    X, y = [], []
    # for i, row in tqdm(df.iterrows(), total=len(df)):
    #     X_temp, y_temp = generate_traindata_for_profile(row.to_dict(),
    #                                 df, num_pairs=per_profile_pairs)
    #     X += X_temp
    #     y += y_temp
    n = len(df)
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(generate_traindata_for_profile,
         range(n), repeat(df, n), repeat(per_profile_pairs, n)), total=n))

    for result in results:
        X += result[0]
        y += result[1]

    X = np.asarray(X)
    y = np.asarray(y)
    print('Training Data Ready. Shape details:')
    print(f'X: {X.shape}, y: {y.shape}')
    return X, y


if __name__ == "__main__":
    df = pd.read_csv('../data/train.csv')
    df = df.fillna('')

    X, y = generate_trainset(df)

    np.save('../data/trainX.npy', X)
    np.save('../data/trainy.npy', y)
