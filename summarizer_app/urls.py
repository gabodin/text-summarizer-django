from django.urls import path

from . import views

urlpatterns = [
    path("", views.summarizer, name="summarizer"),
    path("summary-result", views.summary_result, name="summary_result"),
]
