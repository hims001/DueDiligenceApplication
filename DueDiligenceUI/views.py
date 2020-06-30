from django.shortcuts import render
from django.http import JsonResponse
from .models import SearchModel
from .forms import SearchForm
from DueDiligenceUI.BusinessLogic.process_articles import SearchProcess
from django.views.decorators.csrf import csrf_protect
from django.utils import timezone
from django.views import View


class SearchView(View):
    """
    Class based view for search view
    """
    template_name = 'search.html'

    def get(self, request):
        form = SearchForm()
        return render(request, self.template_name, {'searchForm': form})

    def post(self, request):
        filled_form = SearchForm(request.POST)
        if filled_form.is_valid():
            # Save search to database
            filled_form.save()
        return render(request, self.template_name, {'searchForm': filled_form})


@csrf_protect
def process_articles(request):
    """
    AJAX endpoint
    :param request:
    :return: JSON output response
    """
    entity = request.POST.get('searchText').strip()
    if request.session.get(entity, None) is None:
        searchModel = SearchModel()
        searchModel.SearchText = entity
        searchModel.RequestedDate = timezone.now()
        searchModel.save(force_insert=True)
        # print(searchModel.Id)
        # Passing search text to model for prediction
        sp = SearchProcess()
        outcome = sp.process_request(entityname=entity, model_id=searchModel.Id)

        # Store entity and outcome in session for repeated requests
        request.session[entity] = outcome
        request.session.set_expiry(120)
    else:
        outcome = SearchProcess.OutputParams(request.session[entity][0],
                                             request.session[entity][1],
                                             request.session[entity][2])

    if outcome.is_success:
        return JsonResponse({
            'outcome': outcome.is_success,
            'probabilityList': outcome.prediction_list,
            'articlesList': outcome.predicted_articles
        })
    else:
        return JsonResponse({'outcome': outcome.is_success})
