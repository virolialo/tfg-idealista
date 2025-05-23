from django.db import models

class Barriada(models.Model):
    id = models.CharField(max_length=255, unique=True, primary_key=True)  # Identificador único
    nombre = models.CharField(max_length=255)  # Nombre de la barriada

    def __str__(self):
        return f"Barriada {self.id} - Name: {self.nombre}"

class Vivienda(models.Model):
    STATUS_CHOICES = [
        ("NEWCONSTRUCTION", "New Construction"),
        ("2HANDRESTORE", "Second Hand - Restore"),
        ("2HANDGOOD", "Second Hand - Good"),
    ]

    id = models.CharField(max_length=255, unique=True, primary_key=True)  # Identificador único
    precio = models.IntegerField()  # Precio de la vivienda
    metros_construidos = models.IntegerField()  # Metros cuadrados construidos
    num_hab = models.IntegerField()  # Número de habitaciones
    num_wc = models.IntegerField()  # Número de baños
    terraza = models.BooleanField(default=False)  # Tiene terraza
    ascensor = models.BooleanField(default=False)  # Tiene ascensor
    aire_acondicionado = models.BooleanField(default=False)  # Tiene aire acondicionado
    parking = models.BooleanField(default=False)  # Tiene plaza de garaje
    orientacion_norte = models.BooleanField(default=False)  # Orientación norte
    orientacion_sur = models.BooleanField(default=False)  # Orientación sur
    orientacion_este = models.BooleanField(default=False)  # Orientación este
    orientacion_oeste = models.BooleanField(default=False)  # Orientación oeste
    trastero = models.BooleanField(default=False)  # Tiene trastero
    armario_empotrado = models.BooleanField(default=False)  # Tiene armario empotrado
    piscina = models.BooleanField(default=False)  # Tiene piscina
    portero = models.BooleanField(default=False)  # Tiene portero
    jardin = models.BooleanField(default=False)  # Tiene jardín
    duplex = models.BooleanField(default=False)  # Es un dúplex
    estudio = models.BooleanField(default=False)  # Es un estudio
    ultima_planta = models.BooleanField(default=False)  # Está en la última planta
    planta = models.IntegerField()  # Planta (0 o -1)
    plantas_edicio_catastro = models.IntegerField()  # Número total de plantas del edificio
    calidad_catastro = models.PositiveSmallIntegerField()  # Calidad de la vivienda (0-9)
    distancia_centro = models.FloatField()  # Distancia al centro de la ciudad (km)
    distancia_metro = models.FloatField()  # Distancia a la parada de metro más cercana (km)
    distancia_blasco = models.FloatField()  # Distancia a la avenida Blasco (km)
    longitud = models.FloatField()  # Coordenada de longitud
    latitud = models.FloatField()  # Coordenada de latitud
    estado = models.CharField(
        max_length=255,
        choices=STATUS_CHOICES,
        default="NEWCONSTRUCTION",
    ) # Estado de la vivienda (nuevo, segunda mano - restaurar, segunda mano - bueno)
    antiguedad = models.IntegerField()  # Antigüedad de la vivienda (en años)
    barrio = models.ForeignKey(
        Barriada,
        to_field='id',
        on_delete=models.PROTECT,
        related_name='viviendas'
    )

    def __str__(self):
        return f"Housing {self.id} - Price: {self.precio} ({self.metros_construidos} m2)"
