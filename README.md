# DueDiligenceFinalUntouched

manage.py migrate

manage.py makemigrations

manage.py runserver

manage.py showmigrations

manage.py createsuperuser

#For laptop code
sql_db_path = C:\\Users\\HIMS\\Downloads\\DueDilWorkspace\\db.sqlite3
training_data_path = C:\\Users\\HIMS\\Downloads\\DueDilWorkspace\\DueDiligenceUI\\Models\\TrainingData.csv

#For office code
#sql_db_path = C:\\Dev\\Work\\DueDiligence_Material\\DueDil_Final\\DueDiligenceRepo\\db.sqlite3
#training_data_path = C:\\Dev\\Work\\DueDiligence_Material\\DueDil_Final\\DueDiligenceRepo\\DueDiligenceUI\\Models\\TrainingData.csv


(duedilenv) C:\Users\HIMS\Downloads\DueDilWorkspace>python C:\\Users\\HIMS\\Downloads\\DueDilWorkspace\\DueDiligenceUI\\BusinessLogic\\train_model.py
(duedilenv) C:\Users\HIMS\Downloads\DueDilWorkspace>python C:\\Dev\\Work\\DueDiligence_Material\\DueDil_Final\\DueDiligenceRepo\\DueDiligenceUI\\BusinessLogic\\train_model.py


delete from DueDiligenceUI_searchmodel;
delete from sqlite_sequence where name='DueDiligenceUI_searchmodel';

delete from DueDiligenceUI_trainingmodel;
delete from sqlite_sequence where name='DueDiligenceUI_trainingmodel';
