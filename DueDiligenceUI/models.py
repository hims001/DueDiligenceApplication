from django.db import models
from datetime import datetime


class SearchModel(models.Model):
    Id = models.AutoField(primary_key=True)
    SearchText = models.CharField(max_length=100, null=False, blank=False)
    RequestedDate = models.DateTimeField(default=datetime.now, null=False, blank=False)
    Outcome = models.BooleanField(default=0, null=False, blank=False)
    Probability = models.DecimalField(default=0, null=False, blank=False, max_digits=5, decimal_places=2)

    def __str__(self):
        return self.SearchText


class TrainingModel(models.Model):
    Id = models.AutoField(primary_key=True)
    SearchModel = models.ForeignKey('SearchModel', on_delete=models.CASCADE)
    ArticleText = models.TextField(null=False, blank=False)
    Outcome = models.BooleanField(null=True)
    TrainingDate = models.DateTimeField(default=datetime.now, null=False, blank=False)
    Url = models.TextField(null=False, blank=True)
    IsTrained = models.BooleanField(default=0, null=False, blank=False)
