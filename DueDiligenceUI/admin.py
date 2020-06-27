from django.contrib import admin
from .models import SearchModel, TrainingModel


@admin.register(SearchModel)
class SearchModelAdmin(admin.ModelAdmin):
    list_display = ['SearchText', 'Outcome', 'RequestedDate', 'Probability']


@admin.register(TrainingModel)
class TrainingModelAdmin(admin.ModelAdmin):
    list_display = ['ArticleText', 'Outcome', 'TrainingDate', 'SearchModel_id', 'IsTrained']
