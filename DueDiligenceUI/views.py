from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.http import Http404
from .models import SearchModel
from .forms import SearchForm
from DueDiligenceUI.BusinessLogic.process_articles import SearchProcess
from django.views.decorators.csrf import csrf_protect
from django.utils import timezone


def search(request):
    """
    Search route
    """
    if request.method == 'POST':
        filled_form = SearchForm(request.POST)
        if filled_form.is_valid():
            # note = f"Your entity {filled_form.cleaned_data['SearchText']} is being analysed..."
            # Save search to database
            filled_form.save()

    form = SearchForm()
    return render(request, 'search.html', {'searchForm': form})


@csrf_protect
def process_articles(request):
    """
    AJAX endpoint
    :param request:
    :return: JSON output response
    """
    entity = request.POST.get('searchText').strip()

    searchModel = SearchModel()
    searchModel.SearchText = entity
    searchModel.RequestedDate = timezone.now()
    searchModel.save(force_insert=True)
    # print(searchModel.Id)
    # Passing search text to model for prediction
    sp = SearchProcess()
    outcome = sp.process_request(entityname=entity, model_id=searchModel.Id)
    # print(outcome)
    if outcome.is_success:
        return JsonResponse({
            'outcome': outcome.is_success,
            'probabilityList': outcome.prediction_list,
            'articlesList': outcome.predicted_articles
        })
    return JsonResponse({'outcome': outcome.is_success})
