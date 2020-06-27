from django import forms
from .models import SearchModel


class SearchForm(forms.ModelForm):
    class Meta:
        model = SearchModel
        fields = ['SearchText']
        labels = {'SearchText': ''}
        widgets = {'SearchText': forms.TextInput(attrs={'placeholder': 'What are you looking for?', "autocomplete": "off"})}