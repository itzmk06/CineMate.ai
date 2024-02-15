from django.contrib import admin
from django.urls import path, include

admin.site.site_header = "CineMate.ai"
admin.site.site_title = "CineMate.ai Admin Portal"
admin.site.index_title = "Welcome to CineMate.ai Admin Portal"

from django.urls import path
from home import views

urlpatterns = [
    path('', views.login, name="login"), 
    path('about/', views.about, name="about"),
    path('index/', views.index, name='index'),
    path('movies_data/', views.movies_data, name='movies_data'),

]

