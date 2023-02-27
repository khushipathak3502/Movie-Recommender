#!/usr/bin/env python
# coding: utf-8

# In[312]:


import pandas as pd
import numpy as np


# In[313]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[314]:


movies.head()


# In[315]:


credits.head()


# In[316]:


credits.head(1)['cast'].values


# In[317]:


movies = movies.merge(credits,on='title')


# In[318]:


movies.head(1)


# In[319]:


movies['original_language'].value_counts()


# In[320]:


movies.info()


# In[321]:


# genres
# id
# keywords
# title
# overview
# cast
# crew


# In[322]:


movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]


# In[323]:


movies.isnull().sum()


# In[324]:


movies.dropna(inplace=True)


# In[325]:


movies.duplicated().sum()


# In[326]:


movies.iloc[0].genres


# In[327]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[328]:


movies['genres'] = movies['genres'].apply(convert)


# In[329]:


movies.head()


# In[330]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[331]:


movies.head()


# In[332]:


movies['cast'][0]


# In[333]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[334]:


movies['cast'] = movies['cast'].apply(convert3)


# In[335]:


movies.head()


# In[336]:


movies['crew'][0]


# In[337]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[338]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[339]:


movies.head()


# In[340]:


movies['overview'][0]


# In[341]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[342]:


movies.head()


# In[343]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[344]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[345]:


movies.head()


# In[346]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[347]:


movies.head()


# In[348]:


new_df = movies[['movie_id', 'title', 'tags']]


# In[349]:


new_df


# In[350]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[351]:


new_df.head()


# In[352]:


new_df['tags'][0]


# In[353]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[354]:


new_df.head()


# In[355]:


import nltk


# In[356]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[357]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[358]:


ps.stem("loved")


# In[359]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[360]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[361]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[362]:


vectors[0]


# In[363]:


cv.get_feature_names()


# In[364]:


from sklearn.metrics.pairwise import cosine_similarity


# In[365]:


cosine_similarity(vectors)


# In[366]:


cosine_similarity(vectors).shape


# In[367]:


similarity = cosine_similarity(vectors)


# In[371]:


similarity[0]


# In[378]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[379]:


recommend('Avatar')


# In[380]:


import pickle


# In[381]:


pickle.dump(new_df,open('Movies.pkl','wb'))


# In[383]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[386]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




