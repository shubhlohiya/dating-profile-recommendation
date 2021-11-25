import argparse
import numpy as np
import pandas as pd
import pickle
from proxy_label_assignment import get_proxy_label
from feature_utils import get_compatibility_feature
from train_data_gen import iscandidate
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat

def get_recommendation(A, df, mode='ml', model=None, topk=5, threshold=0.25):
    # A = df.iloc[i].to_dict()
    matches = df[df.apply(lambda row: iscandidate(A, row.to_dict()), axis=1)]
    if len(matches) > 100:
        matches = matches.sample(100)  # restrict candidate pool

    X, y = [], []
    for i, row in matches.iterrows():
        X.append(get_compatibility_feature(A, row.to_dict()))
        y.append(get_proxy_label(A, row.to_dict()))
    X, y = np.asarray(X), np.asarray(y)

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
    _, score = get_recommendation(
        test_df.iloc[i].to_dict(), profiles_df, mode, model, topk, threshold)
    return score


def eval_test_recos(test_df, profiles_df, mode='ml', model=None,
                     topk=5, threshold=0.25, multiprocessing=True):

    model = pickle.load(
        open(f'{av.model}-model.pkl', 'rb')) if mode == 'ml' else None

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
                                                repeat(topk, n), repeat(threshold, n)), total=n))
    return np.mean(rel_scores)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=False, default='ml')
    ap.add_argument("--model", required=False, default='logistic')
    ap.add_argument("--topk", required=False, type=int, default=5)
    ap.add_argument("--threshold", required=False, type=float, default=0.25)
    ap.add_argument("--samplesize", required=False, type=int, default=100)

    av = ap.parse_args()

    test_df = pd.read_csv('../data/test.csv')
    profiles_df = pd.read_csv('../data/train.csv')

    test_df = test_df.fillna('')
    profiles_df = profiles_df.fillna('')

    test_df = test_df.sample(100)

    eval_score = eval_test_recos(test_df, profiles_df, **av.__dict__)

    print(f'Recommendation Relevancy Score: {eval_score:.4f}')
