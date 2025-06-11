from django.contrib import admin
from .models import Vivienda, Barriada, Metro, Hiperparametro

admin.site.register(Vivienda)
admin.site.register(Barriada)
admin.site.register(Metro)
admin.site.register(Hiperparametro)