from django.shortcuts import render
from .forms import ViviendaPrediccionForm
from .utils import (
    procesar_viviendas_barrio,
    crear_grafo_vecindad,
    entrenar_gnn_optuna,
    predecir_precio_vivienda,
    generar_id_vivienda_unico,
    obtener_barrio_desde_geojson,
    preparar_datos_vivienda_form,
    calcular_distancias,
    obtener_features_ordenados,

)
from .models import Barriada
import numpy as np

def prediccion_vivienda_view(request):
    precio_predicho = None
    if request.method == 'POST':
        form = ViviendaPrediccionForm(request.POST)
        if form.is_valid():
            # Se obtienen los datos del formulario
            datos = form.cleaned_data

            # Se procesan los datos de la vivienda
            resultado = preparar_datos_vivienda_form(datos)
            print(f"Datos procesados de la vivienda: {resultado}")

            # Se generan atributos generados con campos del formulario
            distancia_metro, distancia_centro, distancia_blasco = calcular_distancias(resultado['latitud'], resultado['longitud'])
            identificador = generar_id_vivienda_unico()

            # Se selecciona el barrio correspondiente a la vivienda del formulario
            barrio = obtener_barrio_desde_geojson(resultado['latitud'], resultado['longitud'])
            resultados, scaler_features = procesar_viviendas_barrio(barrio)
            grafo = crear_grafo_vecindad(resultados)
            precios_reales = np.array([v['precio'] for v in resultados]).reshape(-1, 1)
            modelo, scaler_precio = entrenar_gnn_optuna(grafo, precios_reales)

            # Prepara los features en el orden correcto
            features_ordenados = obtener_features_ordenados(resultado, distancia_metro, distancia_centro, distancia_blasco, identificador)

            # Extrae las coordenadas de la vivienda a predecir
            vivienda_coord = [features_ordenados['latitud'], features_ordenados['longitud']]

            # Prepara la lista de features para el modelo (excluyendo id, precio, latitud, longitud, barrio)
            campos_excluir = ('id', 'precio', 'latitud', 'longitud', 'barrio')
            vivienda_features = [features_ordenados[k] for k in features_ordenados if k not in campos_excluir]

            # Realiza la predicción
            precio_predicho = predecir_precio_vivienda(
                modelo,
                grafo,
                vivienda_features,
                vivienda_coord,
                scaler_precio,
                scaler_features
            )

            if precio_predicho is not None:
                # Redondea a la centena más cercana inferior (por ejemplo, 28540 -> 28500)
                precio_predicho = int(precio_predicho // 100 * 100)
    else:
        form = ViviendaPrediccionForm()

    return render(request, 'prediccion_vivienda.html', {'form': form, 'precio_predicho': precio_predicho})