
# This is a form of unsupervised learning
# There are three types of recommendation system
# content based -oldest type
# colaborative filtering -newest type
# hybrid of above

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

movies=pd.read_csv("/content/drive/MyDrive/College Projects/dataset/tmdb_5000_movies.csv")
credits=pd.read_csv("/content/drive/MyDrive/College Projects/dataset/tmdb_5000_credits.csv")

movies.head(1)

movies.shape

credits.head(1)

credits.shape

# we have to merge to data frames
# pandas have the merge method which merges two dataframes
movies=movies.merge(credits,left_on="id",right_on="movie_id")

movies.head(1)

movies.shape

# to get all the columns present in the dataset we can make use of dataframe.columns

movies.columns

#we are building content based recommender system by taking
# genres,id,keywords,overview,title_x,cast,crew

movies=movies[["genres","movie_id","keywords","overview","title_x","cast","crew"]]

movies.head(1)

movies.shape

# we can rename movie title using rename method pf pandas
movies.rename(columns={"title_x":"title"},inplace=True)

movies.head(1)

movies.isna().sum()

movies.dropna(inplace=True)

movies.isna().sum()

movies.duplicated().sum()

movies.sample()

movies.iloc[568].genres

# ast we will be using since the obj is str, but we need list-> to be more specific ast.literal_eval(str_obj)
import ast
def convert(list_obj):
  L=[]
  for i in ast.literal_eval(list_obj):
    L.append(i["name"])
  return L

movies["genres"]=movies["genres"].apply(convert)

movies.head(4)

movies["keywords"]=movies["keywords"].apply(convert)

movies.head(4)

def convert3(strObj):
  counter=0
  L=[]
  for i in ast.literal_eval(strObj):
    if counter!=3:
      L.append(i["name"])
      counter+=1
    else:
      break
  return L

movies["cast"]=movies["cast"].apply(convert3)

movies.head(3)



def takeDirector(list_obj):
  L=[]
  for i in ast.literal_eval(list_obj):
    if i["job"]=="Director":
      L.append(i["name"])
      break
  return L

movies["crew"]=movies["crew"].apply(takeDirector)

movies.head(4)

movies.sample(4)

movies["genres"]=movies["genres"].apply(lambda x: [i.replace(" ","")for i in x])
movies["keywords"]=movies["keywords"].apply(lambda x: [i.replace(" ","")for i in x])
movies["cast"]=movies["cast"].apply(lambda x: [i.replace(" ","")for i in x])
movies["crew"]=movies["crew"].apply(lambda x: [i.replace(" ","")for i in x])

movies["overview"]=movies["overview"].apply(lambda x: x.split())

movies["tags"]=movies["overview"]+movies["genres"]+movies["cast"]+movies["crew"]+movies["keywords"]

new_data=movies[["movie_id","title","tags"]]

new_data.sample(5)

new_data["tags"]=new_data["tags"].apply(lambda x: " ".join(x))

new_data.head()

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(stop_words='english',max_features=5000)

vectors=cv.fit_transform(new_data['tags']).toarray()

vectors

cv.get_feature_names_out()[0:200]

import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lm=WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def stemming(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)

new_data['tags']=new_data['tags'].apply(stemming)

vectors=cv.fit_transform(new_data['tags']).toarray()

cv.get_feature_names_out()[0:200]

vectors.shape

from sklearn.metrics.pairwise import cosine_similarity

similarity=cosine_similarity(vectors)

similarity.shape

sorted(list(enumerate(similarity[0])),key=lambda x:x[1],reverse=True)[1:11]

def recommend(movie):
  index=new_data[new_data['title']==movie].index[0]
  distances=sorted(list(enumerate(similarity[index])),key=lambda x:x[1],reverse=True)
  for i in distances[1:11]:
    print(new_data.iloc[i[0]].title)

new_data[new_data['title']=="Alice in Wonderland"].index[0]

recommend("Harry Potter and the Half-Blood Prince")



import pickle

pickle.dump(new_data,open('movie_list.pkl','wb'))
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


