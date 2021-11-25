import argparse
import numpy as np
import pandas as pd
import pickle
from proxy_label_assignment import get_proxy_label
from feature_utils import get_compatibility_feature
from train_data_gen import iscandidate
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


def get_recommendation(A, df, mode='ml', model=None, topk=5, threshold=0.25):
    """Returns recommended dating profiles for input user A from database df.

    Args:
        A (dict): Input user profile
        df (pandas.DataFrame): database of dating profiles
        mode (str, optional): Choose from ['naive', 'ml']. Defaults to 'ml'.
            'naive' calculates score by simple averaging.
            'ml' calculates scores by employing specified machine learning model
        model (optional): ML classification model to be used. Defaults to None.
            Sklearn classification models are accepted. Eg. [LogisticRegression,
            LinearSVC, MLPClassifer]
        topk (int, optional): Number of top recommendations to return. Defaults to 5.
        threshold (float, optional): relevance threshold in [0,1] range. Defaults to 0.25.

    Returns:
        (list[dict], list[int]): (topk profiles, boolean relevancy score)
    """
    # A = df.iloc[i].to_dict()
    matches = df[df.apply(lambda row: iscandidate(A, row.to_dict()), axis=1)]
    if len(matches) > 100:
        matches = matches.sample(100)  # restrict candidate pool
    elif len(matches) == 0:
        return None, None
    X, y = [], []
    for i, row in matches.iterrows():
        X.append(get_compatibility_feature(A, row.to_dict()))
        y.append(get_proxy_label(A, row.to_dict()))
    X, y = np.asarray(X), np.asarray(y)
    X.reshape((-1, X.shape[-1]))

    if mode == 'naive':
        scores = np.mean(X, axis=-1)
    elif mode == 'ml':
        assert model is not None, "In ML mode, trained model must be provided."
        scores = model.predict_proba(X)[:, 1]

    reco_indices = (-scores).argsort()[:topk]
    reco_profiles = df.iloc[reco_indices].to_dict('records')
    reco_scores, reco_y = scores[reco_indices], y[reco_indices]

    reco_relevance = int(((1*(reco_scores >= threshold) @ reco_y) /
                          len(reco_y)) >= 0.6) if len(scores) > topk else 1

    return reco_profiles, reco_relevance


def process(i, test_df, profiles_df, mode, model, topk, threshold):
    """Utility function for using multiprocessing for evaluation."""
    _, score = get_recommendation(
        test_df.iloc[i].to_dict(), profiles_df, mode, model, topk, threshold)
    return score


def eval_test_recos(test_df, profiles_df, mode='ml', model=None,
                    topk=5, threshold=0.25, multiprocessing=True):
    """Evaluates the recommendation model on test_df using profiles_df as
    the database of dating profiles.

    Args:
        test_df (pandas.DataFrame): test set dating profiles
        profiles_df (pandas.DataFrame): database of dating profiles
        mode (str, optional): Choose from ['naive', 'ml']. Defaults to 'ml'.
            'naive' calculates score by simple averaging.
            'ml' calculates scores by employing specified machine learning model
        model (optional): ML classification model to be used. Defaults to None.
            Sklearn classification models are accepted. Eg. [LogisticRegression,
            LinearSVC, MLPClassifer]
        topk (int, optional): Number of top recommendations to return. Defaults to 5.
        threshold (float, optional): relevance threshold in [0,1] range. Defaults to 0.25.
        multiprocessing (bool, optional): Indicates if multiprocessing should be used.
            Defaults to True.

    Returns:
        float: Average recommendation performance on test_df.
    """
    model = pickle.load(
        open(f'{model}-model.pkl', 'rb')) if mode == 'ml' else None

    if not multiprocessing:
        rel_scores = []  # container for test relevancy scores
        for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
            _, score = get_recommendation(
                row.to_dict(), profiles_df, mode, model, topk, threshold)
            rel_scores.append(score)

    else:
        n = len(test_df)
        with ProcessPoolExecutor() as executor:  # use multiprocessing to speed up
            rel_scores = list(tqdm(executor.map(process, range(n), repeat(test_df, n),
                                                repeat(profiles_df, n), repeat(
                                                    mode, n), repeat(model, n),
                                                repeat(topk, n), repeat(threshold, n)),
                                                total=n, leave=False))
    rel_scores = [score for score in rel_scores if score is not None]
    return np.mean(rel_scores)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=False, default='ml')
    ap.add_argument("--model", required=False, default='logistic')
    ap.add_argument("--topk", required=False, type=int, default=5)
    ap.add_argument("--threshold", required=False, type=float, default=0.25)
    ap.add_argument("--testsize", required=False, type=int, default=None)

    av = ap.parse_args()

    test_df = pd.read_csv('../data/test.csv')
    profiles_df = pd.read_csv('../data/train.csv')

    test_df = test_df.fillna('')
    profiles_df = profiles_df.fillna('')

    test_df = test_df.sample(av.testsize) if av.testsize else test_df
    del av.testsize

    eval_score = eval_test_recos(test_df, profiles_df, **av.__dict__)

    print(f'Recommendation Relevancy Score: {eval_score:.4f}')
