import numpy as np
from collections import defaultdict
import math
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
# nltk.download('stopwords')


def compatibility_score_naive(X, Y):
    """Does naive computation of compatibility score as mean of the compatibility feature vector."""
    return np.mean(get_compatibility_feature(X, Y))


def clean_and_process_text(tokens, stemmer=PorterStemmer(),
                           stop_words=stopwords.words('english')):
    """Applies stemming and casefolding on text, removes stopwords."""
    tokens = [token.lower() for token in tokens]  # casefolding
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens


def ageFeature(x, y):
    "Returns compatibility feature for the age attributes of x and y"
    age_diff = abs(int(x) - int(y))
    if age_diff < 3:
        feat = 1
    elif age_diff < 5:
        feat = 1/(age_diff+1)
    else:
        feat = 0
    return feat


def bodyTypeFeature(x, y):
    """Gives higher weight to body types that are closer to each other
    based on position on the spectrum of body-types of x and y."""
    def bodyTypeToNum(x):  # All such functions are heuristics based.
        try: x = x.lower()
        except: pass
        if re.search("thin|skinny", x):
            num_x = 3
        elif re.search("average|fit", x):
            num_x = 4
        elif re.search("athletic|jacked", x):
            num_x = 5
        elif re.search("overweight|a little extra|curvy|full figured", x):
            num_x = 3
        elif re.search("used up", x):
            num_x = 2
        else:
            num_x = 0
        return num_x

    num_x, num_y = bodyTypeToNum(x), bodyTypeToNum(y)
    # normalizing using max to restrict 0 to 1 feature weight
    return np.log10(num_x*num_y+1)/np.log10(26)


def dietFeature(x, y):
    """Gives higher weight to similar diets of x and y."""
    def dietToNum(x):
        try: x = x.lower()
        except: pass
        if re.search("vegetarian|vegan", x):
            num_x = 2
        elif re.search("kosher", x):
            num_x = 7
        elif re.search("halal", x):
            num_x = 10
        elif x is not None:
            num_x = 5
        else:  # no response case
            num_x = 0
        return num_x

    if x == y:  # perfect match
        return 1
    else:
        num_x, num_y = dietToNum(x), dietToNum(y)
        return 1/(abs(num_x-num_y)+1) if num_x != 0 and num_y != 0 else 0


def drinkFeature(x, y):
    """Gives higher weight to similar drinking habits of x and y."""
    def drinkToNum(x):
        try: x = x.lower()
        except: pass
        if re.search("often|desperate|playmate|present", x):
            num_x = 1
        elif re.search("social", x):
            num_x = 2
        elif re.search("rare", x):
            num_x = 3
        elif re.search("no|never", x):
            num_x = 4
        else:
            num_x = 0
        return num_x

    if x == y:
        return 1
    else:
        num_x, num_y = drinkToNum(x), drinkToNum(y)
        return 1/(abs(num_x-num_y)+1) if num_x != 0 and num_y != 0 else 0


def drugFeature(x, y):
    """Gives higher weight to similar drug usage of x and y."""
    def drugToNum(x):
        try: x = x.lower()
        except: pass
        if re.search("often", x):
            num_x = 1
        elif re.search("sometimes", x):
            num_x = 2
        elif re.search("never", x):
            num_x = 3
        else:
            num_x = 0
        return num_x

    if x == y:
        return 1
    else:
        num_x, num_y = drugToNum(x), drugToNum(y)
        return 1/(abs(num_x-num_y)+1) if num_x != 0 and num_y != 0 else 0


def educationFeature(x, y):
    """Gives higher weight to similary educated pairs of x and y."""
    def educationToNum(x):
        try: x = x.lower()
        except: pass
        if re.search("space|high school", x):
            num_x = 1
        elif re.search("college", x):
            num_x = 2
        elif re.search("univ|master", x):
            num_x = 3
        elif re.search("law|med|ph", x):
            num_x = 4
        else:
            num_x = 0
        return num_x

    if x == y:
        return 1
    else:
        num_x, num_y = educationToNum(x), educationToNum(y)
        return 1/(abs(num_x-num_y)+1) if num_x != 0 and num_y != 0 else 0


def religionFeature(x, y):
    """Gives higher weight to pairs of x and y with similar religious beliefs."""
    if x == y:
        feat = 1
    elif x != 'atheism' and y != 'atheism':
        a, b = set(x.split()), set(y.split())
        feat = len(a & b) / len(a | b)  # jaccard similarity of tokens
    elif (x == 'atheism' and y != 'atheism') or (y == 'atheism' and x != 'atheism'):
        feat = 1/10
    else:
        feat = 0
    return feat


def smokeFeature(x, y):
    """Gives higher weight to similar smoking habits of x and y."""
    def smokeToNum(x):
        try: x = x.lower()
        except: pass
        if re.search("yes", x):
            num_x = 1
        elif re.search("sometime|drinking|trying to quit", x):
            num_x = 2
        elif re.search("no|never", x):
            num_x = 3
        else:
            num_x = 0
        return num_x

    if x == y:
        return 1
    else:
        num_x, num_y = smokeToNum(x), smokeToNum(y)
        return 1/(abs(num_x-num_y)+1) if num_x != 0 and num_y != 0 else 0


def essaysFeature(x, y, tokenizer=nltk.RegexpTokenizer(r"\w+")):
    """Returns max of cosine and jaccard similarity between input essays."""
    def check_tokenization(x):
        if isinstance(x, str):
            return tokenizer.tokenize(x)
        elif isinstance(x, list):
            return x
        else:
            raise ValueError('Invalid essay data type')

    x, y = check_tokenization(x), check_tokenization(y)
    x, y = clean_and_process_text(x), clean_and_process_text(y)
    dict_x, dict_y = defaultdict(int), defaultdict(int)

    for token in x:
        dict_x[token] += 1
    for token in y:
        dict_y[token] += 1

    mag_x = np.linalg.norm(np.asarray(list(dict_x.values())))
    mag_y = np.linalg.norm(np.asarray(list(dict_y.values())))

    x_tokens, y_tokens = set(dict_x.keys()), set(dict_y.keys())
    union_tokens = x_tokens | y_tokens

    cosine_score = 0.0
    for token in union_tokens:
        cosine_score += dict_x[token]*dict_y[token]
    cosine_score = cosine_score / (mag_x*mag_y) \
         if mag_x != 0 and mag_y != 0 else 0
    jaccard_score = len(x_tokens & y_tokens) / len(union_tokens) \
        if len(union_tokens) != 0 else 0

    return max(cosine_score, jaccard_score)


def petFeature(x, y):
    """Calculates similarity in pet preferences of x and y."""

    if not isinstance(x, str) or not isinstance(y, str):
        return 0

    def get_pet_sentiments(x):
        x_cat, x_dog = 0, 0
        if 'likes cats' in x:
            x_cat = 1
        elif 'dislikes cats' in x:
            x_cat = -1
        if 'likes dogs' in x:
            x_dog = 1
        elif 'dislikes dogs' in x:
            x_dog = -1

        return x_cat, x_dog

    x_cat, x_dog = get_pet_sentiments(x)
    y_cat, y_dog = get_pet_sentiments(y)

    pet_score = (x_cat == y_cat) + (x_dog == y_dog)

    return pet_score


def get_feature_funcs():
    """Returns a dict of (attr, feature_func)."""
    funcs = {
        'age': ageFeature,
        'body_type': bodyTypeFeature,
        'diet': dietFeature,
        'drinks': drinkFeature,
        'drugs': drugFeature,
        'education': educationFeature,
        'religion': religionFeature,
        'essay0': essaysFeature,
        'essay1': essaysFeature,
        'essay2': essaysFeature,
        'essay3': essaysFeature,
        'essay4': essaysFeature,
        'essay5': essaysFeature,
        'essay6': essaysFeature,
        'essay7': essaysFeature,
        'essay8': essaysFeature,
        'essay9': essaysFeature
    }
    return funcs


def get_compatibility_feature(X, Y, feature_funcs=get_feature_funcs()):
    """Generates a compatibility feature vector for Users X and Y.

    Args:
        X (dict): dating profile of user X filtered for recommendation
        Y (dict): dating profile of user Y filtered for recommendation

    Returns:
        ndarray: feature vector of X and Y 's compatibility.
    """

    assert set(X.keys()) == set(
        Y.keys()), "both users should have the same feature keys."
    compat_feature = []

    for key in X:
        if key in feature_funcs:
            compat_feature.append(feature_funcs[key](X[key], Y[key]))

    return np.asarray(compat_feature)
