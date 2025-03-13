from django.urls import path
from . import views

urlpatterns = [
    path('', views.myrun, name = 'main'),
]