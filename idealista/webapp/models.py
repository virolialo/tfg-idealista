from django.db import models

class Barriada(models.Model):
    id = models.CharField(max_length=255, unique=True, primary_key=True)
    nombre = models.CharField(max_length=255)

    def __str__(self):
        return f"Barriada {self.id} - Name: {self.nombre}"

class Hiperparametro(models.Model):
    barrio = models.ForeignKey(
        Barriada,
        to_field='id',
        on_delete=models.CASCADE,
        related_name='hiperparametros'
    )
    hidden_channels = models.IntegerField()
    num_layers = models.IntegerField()
    dropout = models.FloatField()
    lr = models.FloatField()
    epochs = models.IntegerField()

    def __str__(self):
        return f"Hiperparametro {self.barrio.nombre} ({self.barrio.id})"

class Vivienda(models.Model):
    STATUS_CHOICES = [
        ("NEWCONSTRUCTION", "Nueva construcción"),
        ("2HANDRESTORE", "Segunda mano - Restaurar"),
        ("2HANDGOOD", "Segunda mano - Bueno"),
    ]

    id = models.CharField(max_length=255, unique=True, primary_key=True)
    precio_m2 = models.FloatField()
    metros_construidos = models.IntegerField()
    num_hab = models.IntegerField()
    num_wc = models.IntegerField()
    terraza = models.BooleanField(default=False)
    ascensor = models.BooleanField(default=False)
    aire_acondicionado = models.BooleanField(default=False)
    parking = models.BooleanField(default=False)
    orientacion_norte = models.BooleanField(default=False)
    orientacion_sur = models.BooleanField(default=False)
    orientacion_este = models.BooleanField(default=False)
    orientacion_oeste = models.BooleanField(default=False)
    trastero = models.BooleanField(default=False)
    armario_empotrado = models.BooleanField(default=False)
    piscina = models.BooleanField(default=False)
    portero = models.BooleanField(default=False)
    jardin = models.BooleanField(default=False)
    duplex = models.BooleanField(default=False)
    estudio = models.BooleanField(default=False)
    ultima_planta = models.BooleanField(default=False)
    planta = models.IntegerField()
    plantas_edicio_catastro = models.IntegerField()
    calidad_catastro = models.PositiveSmallIntegerField()
    distancia_centro = models.FloatField()
    distancia_metro = models.FloatField()
    distancia_blasco = models.FloatField()
    longitud = models.FloatField()
    latitud = models.FloatField()
    estado = models.CharField(
        max_length=255,
        choices=STATUS_CHOICES,
        default="NEWCONSTRUCTION",
    )
    antiguedad = models.IntegerField()
    barrio = models.ForeignKey(
        Barriada,
        to_field='id',
        on_delete=models.PROTECT,
        related_name='viviendas'
    )

    def __str__(self):
        return f"Vivienda {self.id} - Precio: {self.precio_m2} x ({self.metros_construidos} m2)"
    
class Metro(models.Model):
    id = models.AutoField(primary_key=True)
    nombre = models.CharField(max_length=255)
    latitud = models.FloatField()
    longitud = models.FloatField()

    def __str__(self):
        return f"Metro {self.id} - {self.nombre} ({self.latitud}, {self.longitud})"