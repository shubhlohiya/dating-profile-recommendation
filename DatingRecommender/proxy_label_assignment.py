import numpy as np
from feature_utils import essaysFeature, smokeFeature, petFeature


def get_proxy_label(X, Y, threshold=0.5):
    """Assings proxy label of compatibility for supervised learning."""
    scores = np.array([
        petFeature(X['pets'], Y['pets']),
        smokeFeature(X['smokes'], Y['smokes']),
        # essay0 is used in both ML and proxy labeling
        essaysFeature(X['essay0'], Y['essay0'])
    ])
    weights = np.array([0.4, 0.3, 0.3])
    score = weights @ scores
    return int(score >= threshold)
