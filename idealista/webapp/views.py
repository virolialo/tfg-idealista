from django.shortcuts import render, redirect
from .forms import ViviendaPrediccionForm
from .utils import (
    procesar_viviendas_barrio,
    obtener_barrio_desde_geojson,
    procesar_nueva_vivienda_formulario,
    calcular_distancias,
    predecir_precio_m2_vivienda,
    obtener_vecinos_vivienda,
)

def prediccion_vivienda_view(request):
    if request.method == 'POST':
        form = ViviendaPrediccionForm(request.POST)
        if form.is_valid():
            datos = form.cleaned_data.copy()
            if 'barrio' in datos:
                barrio = datos['barrio']
                datos['barrio_nombre'] = str(barrio)
                datos['barrio_id'] = getattr(barrio, 'id', None)
                del datos['barrio']
            else:
                barrio = obtener_barrio_desde_geojson(datos['latitud'], datos['longitud'])
            # Tratamiento de datos de las viviendas del barrio con el que se predice
            viviendas_procesadas, scaler_features, scaler_target = procesar_viviendas_barrio(barrio)
            # Calculo de distancias 
            distancia_metro, distancia_centro, distancia_blasco = calcular_distancias(
                datos['latitud'], datos['longitud']
            )
            # Tratamiento vivienda a predecir
            vivienda_proc = procesar_nueva_vivienda_formulario(
                datos,
                distancia_blasco,
                distancia_metro,
                distancia_centro,
                scaler_features
            )
            # Prediccion
            precio_predicho = predecir_precio_m2_vivienda(
                viviendas_procesadas,
                vivienda_proc,
                barrio.id,
                scaler_target
            )
            # Resultados de la prediccion
            if precio_predicho is not None:
                metros = datos['metros_construidos']
                precio_m2_predicho = precio_predicho
                precio_total = precio_predicho * metros
                precio_predicho = int(precio_total // 100 * 100)
            else:
                precio_m2_predicho = None
            # Vecinos de la vivienda predicha
            vecinos = obtener_vecinos_vivienda(barrio.id, vivienda_proc)
            request.session['resultado_prediccion'] = {
                'datos': datos,
                'precio_predicho': precio_predicho,
                'precio_m2_predicho': precio_m2_predicho,
                'distancia_blasco': distancia_blasco,
                'distancia_metro': distancia_metro,
                'distancia_centro': distancia_centro,
                'vecinos': [
                    {
                        'latitud': v['latitud'],
                        'longitud': v['longitud'],
                        'precio_m2': v.get('precio_m2', None)
                    } for v in vecinos
                ],
            }
            return redirect('resultado_prediccion')
    else:
        form = ViviendaPrediccionForm()
    return render(request, 'prediccion_vivienda.html', {'form': form})

def resultado_prediccion_view(request):
    resultado = request.session.get('resultado_prediccion')
    if not resultado:
        return redirect('prediccion_vivienda')
    return render(request, 'resultado_prediccion.html', resultado)