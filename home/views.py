from django.shortcuts import render,HttpResponse
from .circular_list import CircularLinkedList

def login(request):
    return render(request,'login.html')
def index(request):
    return render(request,'index.html')
def about(request):
    return HttpResponse("This is about page!")
from django.http import JsonResponse
from .circular_list import CircularLinkedList

def movies_data(request):
    movie_list = CircularLinkedList()
    
    movie_list.insert({
        "title": "Avengers: Age of Ultron",
        "description": '"Age of Ultron" (abbreviated AU) is a 2013 comic book fictional crossover storyline published by Marvel Comics that involved the conquest of the Earth by the sentient robot tyrant Ultron. The storyline consisted of an eponymous, 10-issue core miniseries, and a number of tie-in books.',
        "imageUrl": "https://image.tmdb.org/t/p/w500/4ssDuvEDkSArWEdyBl2X5EHvYKU.jpg",
        "watchUrl": "#"
    })
    
    movie_list.insert({
        "title": "Rush Hour 2",
        "description": "Rush Hour 2 is a 2001 American action comedy movie. It stars Jackie Chan, Chris Tucker, John Lone, and Ziyi Zhang. Brett Ratner directed the film, which Jeff Nathanson wrote.",
        "imageUrl": "https://image.tmdb.org/t/p/w500/nmllsevWzx7XtrlARs3hHJn5Pf.jpg",
        "watchUrl": "#"
    })
    
    movie_list.insert({
        "title": "RoboCop",
        "description": "A 1987 American science fiction action film directed by Paul Verhoeven. The film is about a police officer named Alex Murphy who is fatally wounded in the line of duty and transformed into a cyborg police officer named RoboCop.",
        "imageUrl": "https://image.tmdb.org/t/p/w500/gM5ql3BKYmHG3WtZ0buKXN7xY8O.jpg",
        "watchUrl": "#"
    })

    movies_data = movie_list.get_movies()
    return JsonResponse(movies_data, safe=False)
