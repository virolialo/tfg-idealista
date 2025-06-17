from django.test import TestCase
from webapp.forms import ViviendaPrediccionForm
from unittest.mock import patch, MagicMock
from django.urls import reverse

class ViviendaPrediccionFormTests(TestCase):
    def setUp(self):
        self.valid_data = {
            'metros_construidos': 80,
            'num_hab': 3,
            'num_wc': 2,
            'planta': 2,
            'plantas_edicio_catastro': 3,
            'calidad_catastro': 5,
            'antiguedad': 10,
            'terraza': False,
            'ascensor': True,
            'aire_acondicionado': False,
            'parking': False,
            'orientacion_norte': True,
            'orientacion_sur': False,
            'orientacion_este': False,
            'orientacion_oeste': False,
            'trastero': False,
            'armario_empotrado': False,
            'piscina': False,
            'portero': False,
            'jardin': False,
            'duplex': False,
            'estudio': False,
            'ultima_planta': False,
            'estado': '2HANDGOOD',
            'latitud': 39.470669,
            'longitud': -0.372466,
        }

    def test_form_valid(self):
        with patch('webapp.forms.obtener_barrio_desde_geojson') as mock_barrio:
            mock_barrio.return_value = 'BarrioTest'
            form = ViviendaPrediccionForm(data=self.valid_data)
            self.assertTrue(form.is_valid())

    def test_banos_mayor_que_habitaciones(self):
        data = self.valid_data.copy()
        data['num_wc'] = 5
        data['num_hab'] = 2
        form = ViviendaPrediccionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('num_wc', form.errors)

    def test_planta_baja_no_ultima(self):
        data = self.valid_data.copy()
        data['planta'] = 0
        data['ultima_planta'] = True
        form = ViviendaPrediccionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('ultima_planta', form.errors)

    def test_duplex_necesita_mas_de_una_planta(self):
        data = self.valid_data.copy()
        data['duplex'] = True
        data['plantas_edicio_catastro'] = 1
        form = ViviendaPrediccionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('duplex', form.errors)

    def test_antiguedad_cero_estado_nuevo(self):
        data = self.valid_data.copy()
        data['antiguedad'] = 0
        data['estado'] = '2HANDGOOD'
        form = ViviendaPrediccionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('estado', form.errors)

    def test_planta_mayor_que_edificio(self):
        data = self.valid_data.copy()
        data['planta'] = 5
        data['plantas_edicio_catastro'] = 3
        form = ViviendaPrediccionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('planta', form.errors)

    def test_ascensor_necesita_edificio_alto(self):
        data = self.valid_data.copy()
        data['ascensor'] = True
        data['plantas_edicio_catastro'] = 2
        form = ViviendaPrediccionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('ascensor', form.errors)

    def test_ultima_planta_debe_coincidir(self):
        data = self.valid_data.copy()
        data['ultima_planta'] = True
        data['planta'] = 2
        data['plantas_edicio_catastro'] = 3
        form = ViviendaPrediccionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('ultima_planta', form.errors)

    def test_planta_igual_edificio_no_ultima(self):
        data = self.valid_data.copy()
        data['planta'] = 3
        data['plantas_edicio_catastro'] = 3
        data['ultima_planta'] = False
        form = ViviendaPrediccionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('ultima_planta', form.errors)

    def test_orientacion_obligatoria(self):
        data = self.valid_data.copy()
        data['orientacion_norte'] = False
        data['orientacion_sur'] = False
        data['orientacion_este'] = False
        data['orientacion_oeste'] = False
        form = ViviendaPrediccionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('__all__', form.errors)

    def test_ubicacion_obligatoria(self):
        data = self.valid_data.copy()
        data['latitud'] = None
        form = ViviendaPrediccionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('__all__', form.errors)

class PrediccionViviendaViewTests(TestCase):
    def setUp(self):
        self.valid_data = {
            'metros_construidos': 80,
            'num_hab': 3,
            'num_wc': 2,
            'planta': 2,
            'plantas_edicio_catastro': 3,
            'calidad_catastro': 5,
            'antiguedad': 10,
            'terraza': False,
            'ascensor': True,
            'aire_acondicionado': False,
            'parking': False,
            'orientacion_norte': True,
            'orientacion_sur': False,
            'orientacion_este': False,
            'orientacion_oeste': False,
            'trastero': False,
            'armario_empotrado': False,
            'piscina': False,
            'portero': False,
            'jardin': False,
            'duplex': False,
            'estudio': False,
            'ultima_planta': False,
            'estado': '2HANDGOOD',
            'latitud': 39.470669,
            'longitud': -0.372466,
        }

    def test_prediccion_vivienda_get(self):
        response = self.client.get(reverse('prediccion_vivienda'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'prediccion_vivienda.html')
        self.assertIn('form', response.context)

    @patch('webapp.forms.obtener_barrio_desde_geojson')
    @patch('webapp.views.procesar_viviendas_barrio')
    @patch('webapp.views.calcular_distancias')
    @patch('webapp.views.procesar_nueva_vivienda_formulario')
    @patch('webapp.views.predecir_precio_m2_vivienda')
    @patch('webapp.views.obtener_vecinos_vivienda')
    def test_prediccion_vivienda_post_valido(
        self, mock_vecinos, mock_pred, mock_proc_nueva, mock_dist, mock_proc_barrio, mock_barrio
    ):
        mock_barrio.return_value = MagicMock(id=1, __str__=lambda self: "BarrioTest")
        mock_proc_barrio.return_value = ([{}], MagicMock(), MagicMock())
        mock_dist.return_value = (0.5, 1.2, 0.8)
        mock_proc_nueva.return_value = {}
        mock_pred.return_value = 2000
        mock_vecinos.return_value = [{'latitud': 39.47, 'longitud': -0.37, 'precio_m2': 2100}]
        response = self.client.post(reverse('prediccion_vivienda'), data=self.valid_data)
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse('resultado_prediccion'), response.url)

    def test_resultado_prediccion_redirect_si_no_hay_resultado(self):
        response = self.client.get(reverse('resultado_prediccion'))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse('prediccion_vivienda'), response.url)

    def test_resultado_prediccion_ok(self):
        session = self.client.session
        session['resultado_prediccion'] = {
            'datos': self.valid_data,
            'precio_predicho': 100000,
            'precio_m2_predicho': 2000,
            'distancia_blasco': 0.8,
            'distancia_metro': 0.5,
            'distancia_centro': 1.2,
            'vecinos': [{'latitud': 39.47, 'longitud': -0.37, 'precio_m2': 2100}],
        }
        session.save()
        response = self.client.get(reverse('resultado_prediccion'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'resultado_prediccion.html')
        self.assertIn('precio_predicho', response.context)