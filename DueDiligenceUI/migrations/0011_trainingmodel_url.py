# Generated by Django 2.2.6 on 2019-12-07 14:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('DueDiligenceUI', '0010_auto_20191021_1322'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingmodel',
            name='Url',
            field=models.TextField(blank=True),
        ),
    ]