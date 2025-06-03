from django.contrib import admin
from .models import Vivienda, Barriada, Hiperparametros, Metro

admin.site.register(Vivienda)
admin.site.register(Barriada)
admin.site.register(Hiperparametros)
admin.site.register(Metro)