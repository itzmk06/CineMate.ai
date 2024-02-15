from django.urls import path
from home import views

urlpatterns = [
    path('', views.index, name="login"),  
    path('about/', views.about, name="about"),
    path('index/', views.manoj, name='index'),  
    path('movies_data/', views.home, name='movies_data'),
]
