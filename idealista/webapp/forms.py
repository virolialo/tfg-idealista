from django import forms
from .utils import obtener_barrio_desde_geojson

class ViviendaPrediccionForm(forms.Form):
    metros_construidos = forms.IntegerField(
        label="Metros construidos",
        min_value=20,
        max_value=1000,
        required=True
    )
    num_hab = forms.IntegerField(
        label="Número de habitaciones",
        min_value=0,
        max_value=90,
        required=True
    )
    num_wc = forms.IntegerField(
        label="Número de baños",
        min_value=0,
        max_value=15,
        required=True
    )
    planta = forms.IntegerField(
        label="Planta",
        min_value=-1,
        max_value=15,
        required=True
    )
    plantas_edicio_catastro = forms.IntegerField(
        label="Plantas edificio (catastro)",
        min_value=0,
        max_value=35,
        required=True
    )
    calidad_catastro = forms.IntegerField(
        label="Calidad catastro (1-10)",
        min_value=1,
        max_value=10,
        required=True
    )
    antiguedad = forms.IntegerField(
        label="Antigüedad (años)",
        min_value=0,
        max_value=500,
        required=True
    )
    terraza = forms.BooleanField(label="Terraza", required=False)
    ascensor = forms.BooleanField(label="Ascensor", required=False)
    aire_acondicionado = forms.BooleanField(label="Aire acondicionado", required=False)
    parking = forms.BooleanField(label="Parking", required=False)
    orientacion_norte = forms.BooleanField(label="Orientación norte", required=False)
    orientacion_sur = forms.BooleanField(label="Orientación sur", required=False)
    orientacion_este = forms.BooleanField(label="Orientación este", required=False)
    orientacion_oeste = forms.BooleanField(label="Orientación oeste", required=False)
    trastero = forms.BooleanField(label="Trastero", required=False)
    armario_empotrado = forms.BooleanField(label="Armario empotrado", required=False)
    piscina = forms.BooleanField(label="Piscina", required=False)
    portero = forms.BooleanField(label="Portero", required=False)
    jardin = forms.BooleanField(label="Jardín", required=False)
    duplex = forms.BooleanField(label="Dúplex", required=False)
    estudio = forms.BooleanField(label="Estudio", required=False)
    ultima_planta = forms.BooleanField(label="Última planta", required=False)
    estado = forms.ChoiceField(
        label="Estado",
        choices=[
            ("NEWCONSTRUCTION", "Nueva construcción"),
            ("2HANDRESTORE", "Segunda mano - Restaurar"),
            ("2HANDGOOD", "Segunda mano - Bueno"),
        ],
        required=True
    )
    latitud = forms.FloatField(
        label="Latitud",
        required=True,
        widget=forms.HiddenInput()
    )

    longitud = forms.FloatField(
        label="Longitud",
        required=True,
        widget=forms.HiddenInput()
    )

    # Añadimos los campos ocultos para recibirlos desde el mapa
    latitud = forms.FloatField(widget=forms.HiddenInput())
    longitud = forms.FloatField(widget=forms.HiddenInput())

    def clean(self):
        cleaned_data = super().clean()
        # Validación estricta de tipo int para los campos numéricos
        int_fields = [
            'metros_construidos', 'num_hab', 'num_wc', 'planta',
            'plantas_edicio_catastro', 'calidad_catastro', 'antiguedad'
        ]
        for field in int_fields:
            value = cleaned_data.get(field)
            if not isinstance(value, int):
                self.add_error(field, "Este campo debe ser un número entero.")

        # Validación de calidad_catastro obligatoria
        calidad = cleaned_data.get('calidad_catastro')
        if calidad in [None, '']:
            raise forms.ValidationError("Debe añadir una valoración en calidad catastro.")

        # Validación estricta de tipo bool para los campos booleanos
        bool_fields = [
            'terraza', 'ascensor', 'aire_acondicionado', 'parking',
            'orientacion_norte', 'orientacion_sur', 'orientacion_este', 'orientacion_oeste',
            'trastero', 'armario_empotrado', 'piscina', 'portero', 'jardin',
            'duplex', 'estudio', 'ultima_planta'
        ]
        for field in bool_fields:
            value = cleaned_data.get(field)
            if value not in [True, False]:
                self.add_error(field, "Este campo debe ser verdadero o falso (sí/no).")

        # Validación de al menos una orientación seleccionada
        orientaciones = [
            cleaned_data.get('orientacion_norte', False),
            cleaned_data.get('orientacion_sur', False),
            cleaned_data.get('orientacion_este', False),
            cleaned_data.get('orientacion_oeste', False),
        ]
        if not any(orientaciones):
            raise forms.ValidationError("Debe marcar al menos una orientación (norte, sur, este u oeste).")

        # Validación de barrio usando latitud y longitud
        lat = cleaned_data.get('latitud')
        lon = cleaned_data.get('longitud')
        if lat is None or lon is None:
            raise forms.ValidationError("Debe seleccionar una ubicación en el mapa.")
        barrio = obtener_barrio_desde_geojson(lat, lon)
        if not barrio:
            raise forms.ValidationError("La ubicación seleccionada no pertenece a ningún barrio registrado.")
        cleaned_data['barrio'] = barrio
        return cleaned_data