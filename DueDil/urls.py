"""DueDil URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from DueDiligenceUI import views
from DueDiligenceUI.views import process_articles

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.search, name='search'),
    path('process_articles/', process_articles, name='process_articles'),
    path('employee/', views.home, name='home'),
    path('employees/<int:id>/', views.employee_detail, name='employee_detail'),
]

admin.site.site_header = "Due Diligence Administration"
admin.site.site_title = "Due Diligence Admin Portal"
admin.site.index_title = "Welcome to Due Diligence Admin Portal"