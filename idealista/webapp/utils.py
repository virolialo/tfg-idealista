from math import radians, sin, cos, sqrt, atan2
from .models import Metro, Barriada, Vivienda
import json
from pathlib import Path
from shapely.geometry import shape, Point
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import optuna
from pathlib import Path
from torch_geometric.data import DataLoader
from django.conf import settings
import torch
from pathlib import Path
from django.conf import settings
from .models import Hiperparametro


def haversine(lat1, lon1, lat2, lon2):
    """
    La funcion calcula la distancia entre dos viviendas
    utilizando la formula del haversine.

    Parametros:
    lat1 (float): Latitud de la primera vivienda.
    lon1 (float): Longitud de la primera vivienda.
    lat2 (float): Latitud de la segunda vivienda.
    lon2 (float): Longitud de la segunda vivienda.

    Returns:
    R * c (float): Distancia entre las dos viviendas en kilometros.

    Raises:
    ValueError: Si las coordenadas no son validas (no son numeros).
    ValueError: Si las coordenadas estan fuera de los limites validos (-90 a 90 para latitudes, -180 a 180 para longitudes).
    """
    R = 6371  # Radio de la Tierra en km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def calcular_distancias(lat, lon):
    """
    La funcion calcula las distancias entre la vivienda dada y:
    - El centro de Valencia (coordenadas fijas).
    - La calle Blasco Ibañez (coordenadas fijas).
    - La parada de metro mas cercana (buscando en la base de datos de Django).

    Parametros:
    lat (float): Latitud de la vivienda.
    lon (float): Longitud de la vivienda.

    Returns:
    min_dist_metro (float): Distancia a la parada de metro mas cercana en kilometros.
    distancia_centro (float): Distancia al centro de Valencia en kilometros.
    distancia_blasco (float): Distancia a la calle Blasco Ibañez en kilometros.

    Raises:
    ValueError: Si las coordenadas no son validas (no son numeros).
    ValueError: Si las coordenadas estan fuera de los limites validos (-90 a 90 para latitudes, -180 a 180 para longitudes).
    """
    # Coordenadas centro Valencia
    centro_lat, centro_lon = 39.48006094163613, -0.3790958147344281

    # Coordenadas avenida de Blasco Ibañez
    blasco_lat, blasco_lon = 39.474950137703814, -0.3510915362788804

    # Calculo de distancias
    distancia_centro = haversine(lat, lon, centro_lat, centro_lon)
    distancia_blasco = haversine(lat, lon, blasco_lat, blasco_lon)
    min_dist_metro = None
    for metro in Metro.objects.all():
        dist = haversine(lat, lon, metro.latitud, metro.longitud)
        if (min_dist_metro is None) or (dist < min_dist_metro):
            min_dist_metro = dist

    return  min_dist_metro, distancia_centro, distancia_blasco

def obtener_barrio_desde_geojson(lat, lon):
    """
    La funcion obtiene el barrio al que pertenece una vivienda
    dado su latitud y longitud, utilizando un archivo GeoJSON
    que contiene la geometria de los barrios.

    Parametros:
    lat (float): Latitud de la vivienda.
    lon (float): Longitud de la vivienda.

    Returns:
    Barriada (django.models): Instancia del modelo Barriada correspondiente al barrio de la vivienda.
    None: Si no se encuentra el barrio correspondiente.

    Raises:
    FileNotFoundError: Si el archivo GeoJSON no se encuentra en la ruta especificada.
    ValueError: Si las coordenadas no son validas (no son numeros).
    ValueError: Si las coordenadas estan fuera de los limites validos (-90 a 90 para latitudes, -180 a 180 para longitudes).
    OSError: Si hay un error al abrir o leer el archivo GeoJSON.
    Barriada.DoesNotExist: Si no se encuentra un barrio con el NEIGHBOURID correspondiente en la base de datos.
    """
    # Ruta al archivo GeoJSON
    base_dir = Path(__file__).resolve().parent.parent
    geojson_path = base_dir / "webapp" / "data" / "barris-barrios.geojson"

    # Crea un punto con las coordenadas de la vivienda
    punto = Point(lon, lat)

    with open(geojson_path, encoding='utf-8') as f:
        geojson_data = json.load(f)

    # Busqueda del barrio
    for feature in geojson_data['features']:
        poligono = shape(feature['geometry'])
        if poligono.contains(punto):
            neighbour_id = feature['properties'].get('NEIGHBOURID')
            if neighbour_id:
                try:
                    return Barriada.objects.get(id=neighbour_id)
                except Barriada.DoesNotExist:
                    return None
    return None

def procesar_viviendas_barrio(barrio):
    """
    La funcion procesa las viviendas de un barrio dado,
    normalizando sus atributos y creando un formato adecuado
    para su uso en un modelo de machine learning.

    Parametros:
    barrio (Barriada): Instancia del modelo Barriada correspondiente al barrio de las viviendas.

    Returns:
    list[dict]: Lista de diccionarios con los atributos normalizados de las viviendas del barrio.
    MinMaxScaler: Objeto de normalizacion para los features.
    MinMaxScaler: Objeto de normalizacion para el target (precio_m2).

    Raises:
    ValueError: Si el barrio no es una instancia del modelo Barriada o no tiene un id valido.
    ValueError: Si no se encuentran viviendas en el barrio.
    ValueError: Si hay un error al procesar los datos de las viviendas.
    ValueError: Si hay un error al normalizar los datos.
    ValueError: Si hay un error al crear la lista de viviendas procesadas.
    """
    # Obtencion de las viviendas del barrio
    barrio_id = barrio.id if hasattr(barrio, 'id') else barrio
    viviendas = Vivienda.objects.filter(barrio_id=barrio_id)

    booleanos = [
        'terraza', 'ascensor', 'aire_acondicionado', 'parking',
        'orientacion_norte', 'orientacion_sur', 'orientacion_este', 'orientacion_oeste',
        'trastero', 'armario_empotrado', 'piscina', 'portero', 'jardin',
        'duplex', 'estudio', 'ultima_planta'
    ]

    features_a_normalizar = [
        'metros_construidos', 'num_hab', 'num_wc', 'planta', 'plantas_edicio_catastro',
        'calidad_catastro', 'distancia_centro', 'distancia_metro', 'distancia_blasco', 'antiguedad'
    ]

    orden = [
        'precio_m2', 'metros_construidos', 'num_hab', 'num_wc', 'terraza', 'ascensor',
        'aire_acondicionado', 'parking', 'orientacion_norte', 'orientacion_sur', 'orientacion_este',
        'orientacion_oeste', 'trastero', 'armario_empotrado', 'piscina', 'portero', 'jardin',
        'duplex', 'estudio', 'ultima_planta', 'planta', 'plantas_edicio_catastro', 'calidad_catastro',
        'distancia_centro', 'distancia_metro', 'distancia_blasco', 'longitud', 'latitud',
        'tipo_1', 'tipo_2', 'tipo_3', 'antiguedad'
    ]

    # Definicion de la lista de viviendas procesadas
    viviendas_lista = []
    for v in viviendas:
        d = {}
        d['precio_m2'] = v.precio_m2
        d['metros_construidos'] = v.metros_construidos
        d['num_hab'] = v.num_hab
        d['num_wc'] = v.num_wc
        d['planta'] = v.planta
        d['plantas_edicio_catastro'] = v.plantas_edicio_catastro
        d['calidad_catastro'] = v.calidad_catastro
        d['distancia_centro'] = v.distancia_centro
        d['distancia_metro'] = v.distancia_metro
        d['distancia_blasco'] = v.distancia_blasco
        d['longitud'] = v.longitud
        d['latitud'] = v.latitud
        d['antiguedad'] = v.antiguedad

        # Transformacion de booleanos a 0/1
        for b in booleanos:
            d[b] = 1 if getattr(v, b) else 0

        # Manejo del atributo estado
        if v.estado == "NEWCONSTRUCTION":
            d['tipo_1'], d['tipo_2'], d['tipo_3'] = 1, 0, 0
        elif v.estado == "2HANDRESTORE":
            d['tipo_1'], d['tipo_2'], d['tipo_3'] = 0, 1, 0
        else:
            d['tipo_1'], d['tipo_2'], d['tipo_3'] = 0, 0, 1

        viviendas_lista.append(d)

    # Normalizacion
    # Atributo objetivo
    precios = np.array([v['precio_m2'] for v in viviendas_lista]).reshape(-1, 1)
    scaler_target = MinMaxScaler()
    precios_norm = scaler_target.fit_transform(precios)
    for i, v in enumerate(viviendas_lista):
        v['precio_m2'] = precios_norm[i, 0]

    # Atributos independientes numericos
    X_features = np.array([[v[f] for f in features_a_normalizar] for v in viviendas_lista])
    scaler_features = MinMaxScaler()
    X_features_norm = scaler_features.fit_transform(X_features)
    for i, v in enumerate(viviendas_lista):
        for j, f in enumerate(features_a_normalizar):
            v[f] = X_features_norm[i, j]

    # Orden de los pares clave-valor
    viviendas_procesadas = []
    for v in viviendas_lista:
        v_ordenado = {k: v[k] for k in orden}
        viviendas_procesadas.append(v_ordenado)
    return viviendas_procesadas, scaler_features, scaler_target

def crear_grafo_vecindad(viviendas_procesadas):
    """
    La funcion crea un grafo de vecindad a partir de una lista de viviendas procesadas,
    conectando aquellas que estan dentro de un radio determinado.

    Parametros:
    viviendas_procesadas (list[dict]): Lista de diccionarios con los atributos de las viviendas procesadas.

    Returns:
    Data: Objeto de tipo Data de PyTorch Geometric que representa el grafo de vecindad.

    Raises:
    ValueError: Si la lista de viviendas procesadas esta vacia.
    ValueError: Si las viviendas procesadas no contienen las claves 'latitud' y 'longitud'.
    ValueError: Si las coordenadas de las viviendas no son validas (no son numeros).
    ValueError: Si las coordenadas de las viviendas estan fuera de los limites validos (-90 a 90 para latitudes, -180 a 180 para longitudes).
    ValueError: Si hay un error al calcular las distancias entre las viviendas.
    """
    n = len(viviendas_procesadas)
    latitudes = np.array([v['latitud'] for v in viviendas_procesadas])
    longitudes = np.array([v['longitud'] for v in viviendas_procesadas])

    # Radio fijo de 50m
    radio_km = 0.05

    # Matriz de distancias
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = haversine(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Aristas del grafo
    edge_index = []
    edge_attr = []
    for i in range(n):
        vecinos = []
        for j in range(n):
            if i != j and dist_matrix[i, j] <= radio_km:
                edge_index.append([i, j])
                edge_attr.append([dist_matrix[i, j]])
                vecinos.append(j)
        # Si no tiene vecinos, conecta con el mas cercano
        if not vecinos:
            nearest = np.argmin(dist_matrix[i] + np.eye(n)[i]*1e6)
            edge_index.append([i, nearest])
            edge_attr.append([dist_matrix[i, nearest]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Atributos de los nodos (features)
    features_keys = [k for k in viviendas_procesadas[0].keys() if k not in ('precio_m2', 'latitud', 'longitud')]
    x = torch.tensor([[v[k] for k in features_keys] for v in viviendas_procesadas], dtype=torch.float)

    # Atributo objetivo (target)
    y = torch.tensor([v['precio_m2'] for v in viviendas_procesadas], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data

class GCNRegressor(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, 1))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr.squeeze() if hasattr(data, "edge_attr") else None
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        return x.squeeze()

def procesar_nueva_vivienda_formulario(datos_formulario, distancia_blasco, distancia_metro, distancia_centro, scaler_features):
    """
    La funcion procesa los datos de una nueva vivienda a partir de un formulario,
    normalizando sus atributos y creando un formato adecuado para su uso en un modelo de machine learning.

    Parametros:
    datos_formulario (dict): Diccionario con los datos de la nueva vivienda.
    distancia_blasco (float): Distancia a la calle Blasco Ibañez en kilometros.
    distancia_metro (float): Distancia a la parada de metro mas cercana en kilometros.
    distancia_centro (float): Distancia al centro de Valencia en kilometros.
    scaler_features (MinMaxScaler): Objeto de normalizacion para los features.

    Returns:
    dict: Diccionario con los atributos normalizados de la nueva vivienda, ordenados segun un esquema predefinido.

    Raises:
    ValueError: Si los datos del formulario no contienen las claves necesarias.
    ValueError: Si los datos del formulario no son validos (no son numeros o booleanos).
    """
    booleanos = [
        'terraza', 'ascensor', 'aire_acondicionado', 'parking',
        'orientacion_norte', 'orientacion_sur', 'orientacion_este', 'orientacion_oeste',
        'trastero', 'armario_empotrado', 'piscina', 'portero', 'jardin',
        'duplex', 'estudio', 'ultima_planta'
    ]

    features_a_normalizar = [
        'metros_construidos', 'num_hab', 'num_wc', 'planta', 'plantas_edicio_catastro',
        'calidad_catastro', 'distancia_centro', 'distancia_metro', 'distancia_blasco', 'antiguedad'
    ]

    orden = [
        'metros_construidos', 'num_hab', 'num_wc', 'terraza', 'ascensor',
        'aire_acondicionado', 'parking', 'orientacion_norte', 'orientacion_sur', 'orientacion_este',
        'orientacion_oeste', 'trastero', 'armario_empotrado', 'piscina', 'portero', 'jardin',
        'duplex', 'estudio', 'ultima_planta', 'planta', 'plantas_edicio_catastro', 'calidad_catastro',
        'distancia_centro', 'distancia_metro', 'distancia_blasco', 'longitud', 'latitud',
        'tipo_1', 'tipo_2', 'tipo_3', 'antiguedad', 'precio_m2'
    ]

    # Definicion del diccionario de la nueva vivienda
    v = {}
    v['metros_construidos'] = datos_formulario['metros_construidos']
    v['num_hab'] = datos_formulario['num_hab']
    v['num_wc'] = datos_formulario['num_wc']
    v['planta'] = datos_formulario['planta']
    v['plantas_edicio_catastro'] = datos_formulario['plantas_edicio_catastro']
    v['calidad_catastro'] = datos_formulario['calidad_catastro']
    v['distancia_centro'] = distancia_centro
    v['distancia_metro'] = distancia_metro
    v['distancia_blasco'] = distancia_blasco
    v['longitud'] = datos_formulario['longitud']
    v['latitud'] = datos_formulario['latitud']
    v['antiguedad'] = datos_formulario['antiguedad']

    # Transformacion de booleanos a 0/1
    for b in booleanos:
        v[b] = 1 if datos_formulario.get(b, False) else 0

    # Manejo del atributo estado
    estado = datos_formulario.get('estado', '2HANDGOOD')
    if estado == "NEWCONSTRUCTION":
        v['tipo_1'], v['tipo_2'], v['tipo_3'] = 1, 0, 0
    elif estado == "2HANDRESTORE":
        v['tipo_1'], v['tipo_2'], v['tipo_3'] = 0, 1, 0
    else:
        v['tipo_1'], v['tipo_2'], v['tipo_3'] = 0, 0, 1

    # Normalizacion de los atributos numericos
    X_features = np.array([[v[f] for f in features_a_normalizar]])
    X_features_norm = scaler_features.transform(X_features)
    for j, f in enumerate(features_a_normalizar):
        v[f] = X_features_norm[0, j]

    # Precio m2 inicializado a 0 (predicho mas adelante )
    v['precio_m2'] = 0

    # Orden final
    v_ordenado = {k: v.get(k, None) for k in orden}
    return v_ordenado

def predecir_precio_m2_vivienda(viviendas_procesadas, vivienda_proc, barrio_id, scaler_target):
    """
    La funcion predice el precio por metro cuadrado de una nueva vivienda
    a partir de un modelo de machine learning, utilizando un grafo de vecindad
    creado a partir de las viviendas procesadas del barrio.

    Parametros:
    viviendas_procesadas (list[dict]): Lista de diccionarios con los atributos de las viviendas procesadas del barrio.
    vivienda_proc (dict): Diccionario con los atributos de la nueva vivienda procesada.
    barrio_id (int): ID del barrio al que pertenece la nueva vivienda.
    scaler_target (MinMaxScaler): Objeto de normalizacion para el target (precio_m2).

    Returns:
    float: Precio por metro cuadrado predicho para la nueva vivienda, desnormalizado y limitado a un rango razonable.

    Raises:
    ValueError: Si las viviendas procesadas estan vacias o no contienen las claves necesarias.
    ValueError: Si la nueva vivienda procesada no contiene las claves necesarias.
    ValueError: Si el barrio_id no es valido o no existe en la base de datos.
    ValueError: Si hay un error al crear el grafo de vecindad o al cargar el modelo.
    ValueError: Si hay un error al predecir el precio_m2 de la nueva vivienda.
    ValueError: Si hay un error al desnormalizar el precio_m2 predicho.
    """

    # 1. Procesar la nueva vivienda y añadirla a la lista de viviendas procesadas
    viviendas_full = viviendas_procesadas + [vivienda_proc]

    # 2. Crear el nuevo grafo con la vivienda añadida
    grafo_nuevo = crear_grafo_vecindad(viviendas_full)

    # 3. Cargar hiperparametros y pesos del modelo
    params = Hiperparametro.objects.get(barrio_id=barrio_id)
    BASE_DIR = Path(settings.BASE_DIR)
    weight_path = BASE_DIR / "webapp" / "data" / "weight" / f"pesos_gcn_{barrio_id}.pt"
    in_channels = grafo_nuevo.x.shape[1]

    # 4. Modelo GCN
    model = GCNRegressor(
        in_channels=in_channels,
        hidden_channels=params.hidden_channels,
        num_layers=params.num_layers,
        dropout=params.dropout
    )
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()

    # 5. Predecir para el ultimo nodo (la nueva vivienda)
    with torch.no_grad():
        out = model(grafo_nuevo)
        pred_norm = out[-1].item()

    # 5. Desnormalizar el resultado
    precio_predicho = scaler_target.inverse_transform([[pred_norm]])[0, 0]

    # 6. Limitar el precio_m2 a un rango razonable
    precio_predicho = max(500, min(precio_predicho, 4000))
    return precio_predicho

def obtener_vecinos_vivienda(barrio_id, vivienda_proc):
    """
    La funcion obtiene los vecinos de una vivienda en un barrio dado,
    utilizando el grafo de vecindad e integrando la nueva vivienda
    en el proceso.

    Parametros:
    barrio_id (int): ID del barrio al que pertenece la nueva vivienda.
    vivienda_proc (dict): Diccionario con los atributos de la nueva vivienda procesada.

    Returns:
    list[dict]: Lista de diccionarios con las latitudes y longitudes de los vecinos de la nueva vivienda.

    Raises:
    ValueError: Si el barrio_id no es valido o no existe en la base de datos.
    ValueError: Si la nueva vivienda procesada no contiene las claves 'latitud' y 'longitud'.
    ValueError: Si las coordenadas de la nueva vivienda no son validas (no son numeros).
    ValueError: Si las coordenadas de la nueva vivienda estan fuera de los limites validos (-90 a 90 para latitudes, -180 a 180 para longitudes).
    ValueError: Si hay un error al procesar las viviendas del barrio o al crear el grafo de vecindad.
    """
    # Radio fijo de 50m
    radio_km = 0.05

    # Integracion de la vivienda nueva en el barrio
    viviendas_procesadas, _, _ = procesar_viviendas_barrio(barrio_id)
    viviendas_full = viviendas_procesadas + [vivienda_proc]

    # Grafo de vecindad
    n = len(viviendas_full)
    latitudes = np.array([v['latitud'] for v in viviendas_full])
    longitudes = np.array([v['longitud'] for v in viviendas_full])

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = haversine(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    idx_vivienda = n - 1
    vecinos_idx = [i for i in range(n-1) if dist_matrix[idx_vivienda, i] <= radio_km]

    if not vecinos_idx and n > 1:
        nearest = np.argmin(dist_matrix[idx_vivienda, :-1])
        vecinos_idx = [nearest]

    # Lista de vecinos de la vivienda
    vecinos = []
    for v in [viviendas_procesadas[i] for i in vecinos_idx]:
        vecinos.append({
            'latitud': float(v['latitud']),
            'longitud': float(v['longitud'])
        })
    return vecinos