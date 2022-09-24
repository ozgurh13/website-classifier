
from utils import ScrapTool, clean_text

import pickle
import pandas as pd


with open('utils.pickle', 'rb') as handle:
    utils = pickle.load(handle)

# unpack dictionary into variables, assigning values to keys
#     model, id_to_category, fitted_vectorizer, df_category_unique
locals().update(utils)


def classify(website_url):
    scrapTool = ScrapTool()

    try:
        website = dict(scrapTool.visit_url(website_url))

    except:
        return None

    text        = clean_text(website['website_text'])
    transformed = fitted_vectorizer.transform([text])
    prediction  = id_to_category[model.predict(transformed)[0]]

    data = pd.DataFrame(
        model.predict_proba(transformed) * 100,
        columns = df_category_unique
    ).T

    data.columns    = ['Probability']
    data.index.name = 'Category'

    probability = data.sort_values(
        ['Probability'],
        ascending = False
    ).apply(lambda x: round(x, 2))

    return prediction, probability


