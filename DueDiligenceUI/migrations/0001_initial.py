# Generated by Django 2.2.4 on 2019-08-27 14:21

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Employee',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('employeeid', models.IntegerField(null=True)),
                ('name', models.CharField(blank=True, max_length=100, null=True)),
            ],
        ),
    ]
