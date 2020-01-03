from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.http import Http404
from .models import Employee, SearchModel
from .forms import SearchForm
from DueDiligenceUI.BusinessLogic.process_articles import SearchProcess
from django.views.decorators.csrf import csrf_protect
from django.utils import timezone


def home(request):
    # return HttpResponse('<b>Hi Hims</b>')
    employees = Employee.objects.all()
    return render(request, 'home.html', {'employees': employees})


def employee_detail(request, id):
    # return HttpResponse(f'<b>details {id}</b>')
    try:
        employee = Employee.objects.get(id=id)
    except Employee.DoesNotExist:
        raise Http404('Employee not found')
    return render(request, 'employee_detail.html', {'employee': employee})

def search(request):
    # return render(request, 'search.html')
    if request.method == 'POST':
        filled_form = SearchForm(request.POST)
        if filled_form.is_valid():
            # note = f"Your entity {filled_form.cleaned_data['SearchText']} is being analysed..."

            # Save search to database
            filled_form.save()

            new_form = SearchForm()
            return render(request, 'search.html', {'searchForm': new_form})  # , 'note': note})
    else:
        form = SearchForm()
        return render(request, 'search.html', {'searchForm': form})


@csrf_protect
def process_articles(request):
    entity = request.POST.get('searchText').strip()

    searchModel = SearchModel()
    searchModel.SearchText = entity
    searchModel.RequestedDate = timezone.now()
    searchModel.save(force_insert=True)
    #print(searchModel.Id)
    # Passing search text to model for prediction
    sp = SearchProcess()
    outcome = sp.process_request(entityname=entity, model_id=searchModel.Id)
    #print(outcome)
    if outcome[0] == 1:
        #model = SearchModel.objects.get(Id=searchModel.Id)
        #model.Outcome = outcome[0]
        #model.Probability = outcome[1]
        #model.save(update_fields=["Outcome", "Probability"])
        return JsonResponse({
                             'outcome': outcome[0],
                             'probabilityList': outcome[1],
                             'articlesList': outcome[2]
                            });
    return JsonResponse({'outcome': outcome[0]})