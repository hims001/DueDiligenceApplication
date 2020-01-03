from django import forms
from .models import SearchModel

# class SearchForm(forms.Form):
#    searchText = forms.CharField(label='Search Text :', max_length=100, widget=forms.Textarea)

class SearchForm(forms.ModelForm):
    class Meta:
        model = SearchModel
        fields = ['SearchText']
        labels = {'SearchText': ''}
        widgets = {'SearchText' : forms.TextInput(attrs={'placeholder': 'What are you looking for?', "autocomplete": "off"})}