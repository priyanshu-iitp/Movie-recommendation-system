import pandas as pd
import numpy as np
import ast



movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')



movies = movies.merge(credits, on='title')

# genres,id,keyword,title,overview,cast,crew
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]



# print(movies.isnull().sum())
movies.dropna(inplace=True)
# print(movies.isnull().sum())

# genres

# print(movies.iloc[0].genres)

 
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)

# print(movies.head())


# keywords
movies['keywords'] = movies['keywords'].apply(convert)

# cast

def convert3(obj):
    L=[]
    count=0
    for i in ast.literal_eval(obj):
        if count!=3:
            L.append(i['name'])
            count+=1
        else:
            break
    return L


movies['cast'] = movies['cast'].apply(convert3)

# crew

def fetch_dir(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l


movies['crew'] = movies['crew'].apply(fetch_dir)


# overview
movies['overview'] = movies['overview'].apply(lambda x:x.split())

# remove space between words
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x ])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x ])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x ])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x ])


#  now make a tag section and concatenate genres+keywodes+cast+crew+overview
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

# make a new data frame that cointain movie_id ,tittle and tags of movies
new_df=movies[['movie_id','title','tags']]
# convert tags item into string
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# convert tags item into lowercase
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: x.lower())


# print(new_df['tags'][0])



# VECTORIZATION


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')

vectors=cv.fit_transform(new_df['tags']).toarray()
# print(vectors[0])


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def  stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df.loc[:, 'tags'] = new_df['tags'].apply(stem)

# COSINE SIMILARITY
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)


# RECOMMEND FUNCTION


def recommend(movie):

    movie_index = new_df[new_df['title'] == movie].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print("\nRecommended Movies:\n")

    for i in movie_list:
        print(new_df.iloc[i[0]].title)


# =========================
# TEST
# =========================

recommend('Batman')

import pickle

pickle.dump(new_df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))


