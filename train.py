
from sklearn                         import metrics
from sklearn.calibration             import CalibratedClassifierCV
from sklearn.linear_model            import RidgeClassifier
from sklearn.model_selection         import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd

import pickle


dataset = pd.read_csv('dataset.csv')
df = dataset[['website_url', 'cleaned_website_text', 'Category']].copy()

df_category_unique = df['Category'].unique()

# create a new column 'category_id' with encoded categories
df['category_id'] = df['Category'].factorize()[0]

# dictionary for translating between prediction and category
category_id_df = df[['Category', 'category_id']].drop_duplicates()
id_to_category = dict(category_id_df[['category_id', 'Category']].values)


tfidf = TfidfVectorizer(
    sublinear_tf = True,
    min_df       = 5,
    ngram_range  = (1, 2),
    stop_words   = 'english'
)

# transform each cleaned_text into a vector
features = tfidf.fit_transform(df.cleaned_website_text).toarray()

labels = df.category_id

X = df['cleaned_website_text']    # collection of text
y = df['Category']                # target or the labels we want to predict

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=1
)

model = RidgeClassifier().fit(X_train, y_train)
y_pred = model.predict(X_test)
print('model accuracy:', metrics.accuracy_score(y_test, y_pred))


fitted_vectorizer        = tfidf.fit(X)
tfidf_vectorizer_vectors = fitted_vectorizer.transform(X)

model_ridge      = RidgeClassifier().fit(tfidf_vectorizer_vectors, df['category_id'])
model_calibrated = CalibratedClassifierCV(
    base_estimator = model_ridge,
    n_jobs         = -1
).fit(tfidf_vectorizer_vectors, df['category_id'])


utils = { 'model'              : model_calibrated
        , 'id_to_category'     : id_to_category
        , 'fitted_vectorizer'  : fitted_vectorizer
        , 'df_category_unique' : df_category_unique }

with open('utils.pickle', 'wb') as handle:
    pickle.dump(utils, handle, protocol=pickle.HIGHEST_PROTOCOL)

