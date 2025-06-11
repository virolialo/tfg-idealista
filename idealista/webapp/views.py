from django.shortcuts import render
from .forms import ViviendaPrediccionForm
from .utils import (
    procesar_viviendas_barrio,
    crear_grafo_vecindad,
    predecir_precio_vivienda_con_pesos,  # <-- Importa la función nueva
    generar_id_vivienda_unico,
    obtener_barrio_desde_geojson,
    preparar_datos_vivienda_form,
    calcular_distancias,
    obtener_features_ordenados,
)
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prediccion_vivienda_view(request):
    precio_predicho = None
    if request.method == 'POST':
        form = ViviendaPrediccionForm(request.POST)
        if form.is_valid():
            datos = form.cleaned_data
            resultado = preparar_datos_vivienda_form(datos)
            distancia_metro, distancia_centro, distancia_blasco = calcular_distancias(resultado['latitud'], resultado['longitud'])
            identificador = generar_id_vivienda_unico()
            barrio = obtener_barrio_desde_geojson(resultado['latitud'], resultado['longitud'])
            resultados, scaler_features = procesar_viviendas_barrio(barrio)
            grafo = crear_grafo_vecindad(resultados)
            precios_reales = np.array([v['precio_m2'] for v in resultados]).reshape(-1, 1)
            
            # Obtén el scaler_precio como lo hacías antes
            scaler_precio = MinMaxScaler()
            scaler_precio.fit(precios_reales)

            features_ordenados = obtener_features_ordenados(resultado, distancia_metro, distancia_centro, distancia_blasco, identificador)
            vivienda_coord = [features_ordenados['latitud'], features_ordenados['longitud']]
            campos_excluir = ('id', 'precio_m2', 'latitud', 'longitud', 'barrio')
            vivienda_features = [features_ordenados[k] for k in features_ordenados if k not in campos_excluir]

            # Prediccion
            precio_predicho = predecir_precio_vivienda_con_pesos(
                barrio,
                grafo,
                vivienda_features,
                vivienda_coord,
                scaler_precio,
                scaler_features
            )

            if precio_predicho is not None:
                print(f"Precio predicho: {precio_predicho}")
                metros = features_ordenados['metros_construidos']
                precio_total = precio_predicho * metros
                precio_predicho = int(precio_total // 100 * 100)
    else:
        form = ViviendaPrediccionForm()

    return render(request, 'prediccion_vivienda.html', {'form': form, 'precio_predicho': precio_predicho})