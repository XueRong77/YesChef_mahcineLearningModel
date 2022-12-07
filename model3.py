import json

import joblib
import pandas as pd
from flask import Flask, request
import nltk
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math

recipes_ingredients = pd.read_csv('recipes_ingredients')
f = open('recipes_corpus.txt', encoding='utf-8')
Recipes_Corpus = f.read()
f.close()

model_03 = joblib.load('tfidf_model3')


def preprocess_data(docs):
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    docs_clean = []
    punc = str.maketrans('', '', string.punctuation)
    for doc in docs:
        doc_no_punc = doc.translate(punc)
        words = doc_no_punc.lower().split()
        words = [lemmatizer.lemmatize(word, 'v')
                 for word in words if word not in stop_words]
        docs_clean.append(' '.join(words))

    return docs_clean


#   userId = req['userId']
ingredients = ['bean chili garlic']
ingredients_query = preprocess_data(ingredients)
print(ingredients_query)

recipes_corpus_docs = Recipes_Corpus.split("', '")
recipes_corpus_docs = preprocess_data(recipes_corpus_docs)

model_03.fit(recipes_corpus_docs)
tfidf_recipes_docs = model_03.transform(recipes_corpus_docs).toarray()
print(tfidf_recipes_docs.shape)


features = model_03.get_feature_names_out()
indexes = [recipes_ingredients.iloc[i, 1]
           for i in range(len(recipes_ingredients))]
#print(indexes)

tfidf_df_recipes = pd.DataFrame(
    data=tfidf_recipes_docs, index=indexes, columns=features)

tfidf_query = model_03.transform(ingredients_query).toarray()
#tfidf_df_query = pd.DataFrame(data=tfidf_query, index='query_ingredient'], columns=features)

docs_similarity = cosine_similarity(tfidf_query, tfidf_df_recipes)
query_similarity = docs_similarity[0]

series = pd.Series(query_similarity, index=tfidf_df_recipes.index)
sorted_series = series.sort_values(ascending=False)
sorted_series = sorted_series[sorted_series != 0]
sorted_id = sorted_series.index
print(sorted_id)

id = []
for e in sorted_id:
    id.append(e)
print(id)

# recommendation_list = sorted_series
# print(recommendation_list[:0])
recommendation_list_json = json.dumps(id)
response_dict = {"prediction": recommendation_list_json}

# recommend_recipes_ingredients = sorted_series[:4].index.to_list()
# recommend_recipes_list = []
# for e in recommend_recipes_ingredients:
#     recommend_recipes_list.append(e)
# response_dic ={"prediction": recommend_recipes_list}
print(response_dict)


#     # objInstance = ObjectId(userId)
#     # target_user = records_users.find_one({'_id': objInstance})
#     # df_target_user = pd.DataFrame(target_user)
#     # df2 = df_target_user['bookmarks'].apply(pd.Series)
#     # df2.drop(columns=['bookmarkDateTime'])
#
#     return str('hi' + userId + ingredient)