# Generated by Django 5.2 on 2025-04-05 23:17

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0004_rename_condition_condition_condition'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.RemoveField(
            model_name='condition',
            name='customer',
        ),
        migrations.CreateModel(
            name='ModelUser',
            fields=[
                ('modeluser_id', models.IntegerField(primary_key=True, serialize=False)),
                ('username', models.CharField(blank=True, max_length=50, null=True, unique=True)),
                ('first_name', models.CharField(blank=True, max_length=100, null=True)),
                ('last_name', models.CharField(blank=True, max_length=100, null=True)),
                ('modeluser_image', models.ImageField(blank=True, null=True, upload_to='')),
                ('password', models.CharField(blank=True, max_length=100, null=True)),
                ('email', models.EmailField(blank=True, max_length=254, null=True)),
                ('phone_number', models.IntegerField(blank=True, help_text='0712345678 or +254712345678', null=True)),
                ('gender', models.CharField(blank=True, choices=[('male', 'Male'), ('female', 'Female')], max_length=10, null=True)),
                ('user', models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'db_table': 'modelUser',
            },
        ),
        migrations.AddField(
            model_name='condition',
            name='model_user',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='main.modeluser'),
        ),
        migrations.DeleteModel(
            name='Customer',
        ),
    ]
