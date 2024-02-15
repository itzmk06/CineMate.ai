import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

movies=pd.read_csv("static/tmdb_5000_credits.csv")
credits=pd.read_csv("static/tmdb_5000_movies .csv")

movies.head(1)

movies.shape

credits.head(1)

credits.shape

movies=movies.merge(credits,left_on="id",right_on="movie_id")

movies.shape

movies.head(1)

movies.columns

# genres,id,keywords,title_x,overview,cast,crew

movies=movies[['movie_id','title_x','overview','genres','keywords','cast','crew']]

movies.shape

movies.head(1)

movies.rename(columns={'title_x':'title'},inplace=True)

movies.head(1)

movies.isnull().sum()

movies.dropna(inplace=True)

movies.shape

movies.isnull().sum()

movies.duplicated().sum()

movies['genres']

movies.sample()

a=movies.iloc[1918].genres
a

type(a)

x={"id": 28, "name": "Action"}

x['id']

import ast

ast.literal_eval("{'a':'b','c':'d','e':'f'}")

ast.literal_eval("[2,4,6,7]")

import ast
def convert(obj):
  L=[]
  for i in ast.literal_eval(obj):
    L.append(i['name'])
  return L

convert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 35, "name": "Comedy"}, {"id": 14, "name": "Fantasy"}]')

movies['genres']=movies['genres'].apply(convert)

movies['genres']

movies.head(3)

movies.iloc[0].keywords

movies['keywords']=movies['keywords'].apply(convert)

movies.sample(4)

movies.iloc[1].cast

def convert3(obj):
  L=[]
  counter=0
  for i in ast.literal_eval(obj):
    if counter !=3:
      L.append(i['name'])
      counter+=1
    else:
      break
  return L

movies['cast']=movies['cast'].apply(convert3)

movies['crew'][0]

def fetch_director(obj):
  L=[]
  for i in ast.literal_eval(obj):
    if i['job']=="Director":
      L.append(i['name'])
      break
  return L

movies['crew']=movies['crew'].apply(fetch_director)

movies.head(10)

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])

movies.head(10)

movies['overview']=movies['overview'].apply(lambda x:x.split())

movies.head(3)

movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

movies.sample()

new_df=movies[['movie_id','title','tags']]

new_df.head()

new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))

new_df.head()

new_df['tags'][8]

from sklearn.feature_extraction.text import CountVectorizer

# To understand count vectorizer

s=['Congrats. You have won a lottery and you can get lottery amount by calling the lottery number',
   'Give your bank account details for lottery amount to be credited to your bank account',
   'lottery for sure if the bank account details are verified']

s[0].split()  # Tokenization

s[1].split()

s[2].split()

vect=CountVectorizer()
op=vect.fit_transform(s).toarray()
op

vect.get_feature_names_out()

import pandas as pd
pd.DataFrame(op,columns=vect.get_feature_names_out(),index=['s[1]','s[2]','s[3]'])

s

# stop_words

# Stop words are the words in a stop list which are filtered out before or after processing of natural language data because they are insignificant.
# Stopwords are the English words which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the sentence.

vect=CountVectorizer(stop_words='english')
op=vect.fit_transform(s).toarray()
pd.DataFrame(op,columns=vect.get_feature_names_out(),index=['s[1]','s[2]','s[3]'])

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(stop_words='english',max_features=5000)

vectors=cv.fit_transform(new_df['tags']).toarray()

vectors

cv.get_feature_names_out()[0:200]

# Stemming
# stemming is the process of reducing inflected words to their root word
# Stemming is a natural language processing technique that is used to reduce words to their base form, also known as the root form

['love','loved','loving','loves']
['love','love','love','love']

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

ps.stem("loves")

def stemming(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)

stemming("love loved loving loves")

stemming("actors actor")

stemming("dancing dance dances danced")

# Lemmatization
import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lm=WordNetLemmatizer()

lm.lemmatize("walking",pos='v')

lm.lemmatize("dancing",pos='v')

new_df['tags']=new_df['tags'].apply(stemming)

vectors=cv.fit_transform(new_df['tags']).toarray()

vectors

cv.get_feature_names_out()[0:200]

# cosine similarity

vectors.shape

from sklearn.metrics.pairwise import cosine_similarity

similarity=cosine_similarity(vectors)

similarity.shape

similarity

sorted(list(enumerate(similarity[0])),key=lambda x:x[1],reverse=True)[1:11]

def recommend(movie):
  index=new_df[new_df['title']==movie].index[0]
  distances=sorted(list(enumerate(similarity[index])),key=lambda x:x[1],reverse=True)
  for i in distances[1:11]:
    print(new_df.iloc[i[0]].title)

new_df[new_df['title']=="Alice in Wonderland"].index[0]

recommend("Harry Potter and the Half-Blood Prince")



import pickle

pickle.dump(new_df,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))

import streamlit as st
import pickle
import requests

movies=pickle.load(open('movie_list.pkl','rb'))
similarity=pickle.load(open('similarity.pkl','rb'))

def fetch_poster(movie_id):
  url='https://api.themoviedb.org/3/movie/{}?api_key=2736a08daef7a534d3cf2d8c371e0427&language=en-US'.format(movie_id)
  data=requests.get(url)
  data=data.json()
  poster_path=data['poster_path']
  full_path="https://image.tmdb.org/t/p/w500"+data['poster_path']
  return full_path

def recommend(movie):
  index=movies[movies['title']==movie].index[0]
  distances=sorted(list(enumerate(similarity[index])),key=lambda x:x[1],reverse=True)
  recommended_movie_names=[]
  recommended_movie_posters=[]

  for i in distances[1:11]:
    movie_id=movies.iloc[i[0]].movie_id
    recommended_movie_names.append(movies.iloc[i[0]].title)
    recommended_movie_posters.append(fetch_poster(movie_id))
  return recommended_movie_names,recommended_movie_posters

st.header("CineMate.ai ")
movie_list=movies['title'].values
selected_movie=st.selectbox("Type or select a movie from the list",movie_list)


if st.button("Show Recommendations"):
  recommended_movie_names,recommended_movie_posters=recommend(selected_movie)
  c1,c2,c3,c4,c5,c6,c7,c8,c9,c10=st.tabs(["Movie 1","Movie 2","Movie 3","Movie 4","Movie 5","Movie 6","Movie 7","Movie 8","Movie 9","Movie 10"])
  with c1:
    st.text(recommended_movie_names[0])
    st.image(recommended_movie_posters[0])
  with c2:
    st.text(recommended_movie_names[1])
    st.image(recommended_movie_posters[1])
  with c3:
    st.text(recommended_movie_names[2])
    st.image(recommended_movie_posters[2])
  with c4:
    st.text(recommended_movie_names[3])
    st.image(recommended_movie_posters[3])
  with c5:
    st.text(recommended_movie_names[4])
    st.image(recommended_movie_posters[4])
  with c6:
    st.text(recommended_movie_names[5])
    st.image(recommended_movie_posters[5])
  with c7:
    st.text(recommended_movie_names[6])
    st.image(recommended_movie_posters[6])
  with c8:
    st.text(recommended_movie_names[7])
    st.image(recommended_movie_posters[7])
  with c9:
    st.text(recommended_movie_names[8])
    st.image(recommended_movie_posters[8])
  with c10:
    st.text(recommended_movie_names[9])
    st.image(recommended_movie_posters[9])



