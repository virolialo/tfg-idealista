from math import radians, sin, cos, sqrt, atan2
from .models import Metro, Barriada, Vivienda
import json
from pathlib import Path
from shapely.geometry import shape, Point
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import optuna
import random
import string

FEATURES_ORDER = [
    'metros_construidos', 'num_hab', 'num_wc', 'terraza', 'ascensor', 'aire_acondicionado', 'parking',
    'orientacion_norte', 'orientacion_sur', 'orientacion_este', 'orientacion_oeste', 'trastero',
    'armario_empotrado', 'piscina', 'portero', 'jardin', 'duplex', 'estudio', 'ultima_planta',
    'calidad_catastro', 'tipo_1', 'tipo_2', 'tipo_3', 'distancia_centro', 'distancia_metro',
    'distancia_blasco', 'antiguedad'
]

def obtener_features_ordenados(resultado, distancia_metro, distancia_centro, distancia_blasco, identificador):
    """
    Devuelve una lista de features ordenados según el orden requerido para la predicción.
    """
    ordered_fields = [
        'id', 'precio', 'metros_construidos', 'num_hab', 'num_wc',
        'terraza', 'ascensor', 'aire_acondicionado', 'parking',
        'orientacion_norte', 'orientacion_sur', 'orientacion_este', 'orientacion_oeste',
        'trastero', 'armario_empotrado', 'piscina', 'portero', 'jardin', 'duplex', 'estudio', 'ultima_planta',
        'calidad_catastro', 'tipo_1', 'tipo_2', 'tipo_3',
        'distancia_centro', 'distancia_metro', 'distancia_blasco',
        'latitud', 'longitud', 'antiguedad', 'barrio'
    ]

    vivienda_dict = {
        'id': identificador,
        'precio_m2': None,  # El precio es el que se va a predecir
        'metros_construidos': resultado.get('metros_construidos'),
        'num_hab': resultado.get('num_hab'),
        'num_wc': resultado.get('num_wc'),
        'terraza': resultado.get('terraza'),
        'ascensor': resultado.get('ascensor'),
        'aire_acondicionado': resultado.get('aire_acondicionado'),
        'parking': resultado.get('parking'),
        'orientacion_norte': resultado.get('orientacion_norte'),
        'orientacion_sur': resultado.get('orientacion_sur'),
        'orientacion_este': resultado.get('orientacion_este'),
        'orientacion_oeste': resultado.get('orientacion_oeste'),
        'trastero': resultado.get('trastero'),
        'armario_empotrado': resultado.get('armario_empotrado'),
        'piscina': resultado.get('piscina'),
        'portero': resultado.get('portero'),
        'jardin': resultado.get('jardin'),
        'duplex': resultado.get('duplex'),
        'estudio': resultado.get('estudio'),
        'ultima_planta': resultado.get('ultima_planta'),
        'calidad_catastro': resultado.get('calidad_catastro'),
        'tipo_1': resultado.get('tipo_1'),
        'tipo_2': resultado.get('tipo_2'),
        'tipo_3': resultado.get('tipo_3'),
        'distancia_centro': distancia_centro,
        'distancia_metro': distancia_metro,
        'distancia_blasco': distancia_blasco,
        'latitud': resultado.get('latitud'),
        'longitud': resultado.get('longitud'),
        'antiguedad': resultado.get('antiguedad'),
        'barrio': resultado.get('barrio').id,
    }

    return vivienda_dict

def preparar_datos_vivienda_form(data):
    """
    Recibe un diccionario con los campos del form y devuelve un nuevo diccionario
    con los campos transformados según las reglas especificadas.
    """
    # Campos booleanos a convertir a 0/1
    bool_fields = [
        'terraza', 'ascensor', 'aire_acondicionado', 'parking', 'orientacion_norte',
        'orientacion_sur', 'orientacion_este', 'orientacion_oeste', 'trastero',
        'armario_empotrado', 'piscina', 'portero', 'jardin', 'duplex', 'estudio', 'ultima_planta'
    ]
    result = data.copy()
    for field in bool_fields:
        result[field] = 1 if data.get(field, False) else 0

    # calidad_catastro: restar 1
    if 'calidad_catastro' in result and result['calidad_catastro'] is not None:
        result['calidad_catastro'] = int(result['calidad_catastro']) - 1

    # estado a tipo_1, tipo_2, tipo_3
    estado = data.get('estado')
    if estado == "NEWCONSTRUCTION":
        result['tipo_1'] = 1
        result['tipo_2'] = 0
        result['tipo_3'] = 0
    elif estado == "2HANDRESTORE":
        result['tipo_1'] = 0
        result['tipo_2'] = 1
        result['tipo_3'] = 0
    elif estado == "2HANDGOOD":
        result['tipo_1'] = 0
        result['tipo_2'] = 0
        result['tipo_3'] = 1
    else:
        result['tipo_1'] = 0
        result['tipo_2'] = 0
        result['tipo_3'] = 0

    # Elimina el campo 'estado' si existe
    if 'estado' in result:
        del result['estado']
    print(result)
    return result

def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia en kilómetros entre dos puntos dados por latitud y longitud usando la fórmula de Haversine.
    """
    R = 6371  # Radio de la Tierra en km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def calcular_distancias(lat, lon):
    """
    Calcula:
    - distancia a la parada de metro más cercana (de la base de datos)
    - distancia al centro de Valencia
    - distancia a la avenida de Blasco Ibáñez

    Devuelve un diccionario con las distancias en km.
    """
    # Coordenadas de referencia
    centro_lat, centro_lon = 39.48006094163613, -0.3790958147344281
    blasco_lat, blasco_lon = 39.474950137703814, -0.3510915362788804

    # Distancia al centro
    distancia_centro = haversine(lat, lon, centro_lat, centro_lon)
    # Distancia a Blasco
    distancia_blasco = haversine(lat, lon, blasco_lat, blasco_lon)

    # Distancia a la parada de metro más cercana
    min_dist_metro = None
    for metro in Metro.objects.all():
        dist = haversine(lat, lon, metro.latitud, metro.longitud)
        if (min_dist_metro is None) or (dist < min_dist_metro):
            min_dist_metro = dist

    return  min_dist_metro, distancia_centro, distancia_blasco

def obtener_barrio_desde_geojson(lat, lon):
    """
    Dadas unas coordenadas, busca en el geojson de barrios (ruta fija)
    y devuelve la instancia de Barriada correspondiente al barrio en el que caen las coordenadas.
    """
    # Ruta fija al geojson de barrios
    base_dir = Path(__file__).resolve().parent.parent
    geojson_path = base_dir / "webapp" / "data" / "barris-barrios.geojson"

    punto = Point(lon, lat)  # GeoJSON usa (lon, lat)

    with open(geojson_path, encoding='utf-8') as f:
        geojson_data = json.load(f)

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

def procesar_viviendas_barrio(barriada):
    """
    Procesa todas las viviendas de un barrio y devuelve una lista de diccionarios
    con los atributos en el orden especificado:
    id, precio, metros_construidos, num_hab, num_wc, terraza, ascensor, aire_acondicionado, parking,
    orientacion_norte, orientacion_sur, orientacion_este, orientacion_oeste, trastero, armario_empotrado,
    piscina, portero, jardin, duplex, estudio, ultima_planta, calidad_catastro, tipo_1, tipo_2, tipo_3,
    distancia_centro, distancia_metro, distancia_blasco, latitud, longitud, antiguedad, barrio
    """
    # Orden de los atributos
    ordered_fields = [
        'id', 'precio_m2', 'metros_construidos', 'num_hab', 'num_wc',
        'terraza', 'ascensor', 'aire_acondicionado', 'parking',
        'orientacion_norte', 'orientacion_sur', 'orientacion_este', 'orientacion_oeste',
        'trastero', 'armario_empotrado', 'piscina', 'portero', 'jardin', 'duplex', 'estudio', 'ultima_planta',
        'calidad_catastro', 'tipo_1', 'tipo_2', 'tipo_3',
        'distancia_centro', 'distancia_metro', 'distancia_blasco',
        'latitud', 'longitud', 'antiguedad', 'barrio'
    ]
    # Booleanos a 0/1
    bool_fields = [
        'terraza', 'ascensor', 'aire_acondicionado', 'parking', 'orientacion_norte',
        'orientacion_sur', 'orientacion_este', 'orientacion_oeste', 'trastero',
        'armario_empotrado', 'piscina', 'portero', 'jardin', 'duplex', 'estudio', 'ultima_planta'
    ]
    # Numéricos a normalizar (excepto precio, id, longitud, latitud)
    num_fields = [
        'metros_construidos', 'num_hab', 'num_wc', 'planta', 'plantas_edicio_catastro',
        'calidad_catastro', 'distancia_centro', 'distancia_metro', 'distancia_blasco',
        'antiguedad'
    ]
    estado_map = {
        "NEWCONSTRUCTION": [1, 0, 0],
        "2HANDRESTORE": [0, 1, 0],
        "2HANDGOOD": [0, 0, 1]
    }

    viviendas_queryset = barriada.viviendas.all()
    if not viviendas_queryset.exists():
        return []

    # Prepara el scaler con los datos del queryset
    X = []
    for v in viviendas_queryset:
        X.append([getattr(v, f) for f in num_fields])
    scaler = MinMaxScaler()
    scaler.fit(X)

    resultados = []
    for vivienda in viviendas_queryset:
        temp = {}
        # id, precio, latitud, longitud, barrio
        temp['id'] = vivienda.id
        temp['precio_m2'] = vivienda.precio_m2
        temp['latitud'] = vivienda.latitud
        temp['longitud'] = vivienda.longitud
        temp['barrio'] = vivienda.barrio.id if hasattr(vivienda.barrio, 'id') else vivienda.barrio

        # Booleanos
        for field in bool_fields:
            temp[field] = int(getattr(vivienda, field))
        # Normaliza los valores numéricos de la vivienda
        v_array = [[getattr(vivienda, f) for f in num_fields]]
        norm_values = scaler.transform(v_array)[0]
        for i, field in enumerate(num_fields):
            temp[field] = norm_values[i]
        # Estado one-hot
        estado = getattr(vivienda, 'estado')
        buildtype = estado_map.get(estado, [0, 0, 0])
        temp['tipo_1'] = buildtype[0]
        temp['tipo_2'] = buildtype[1]
        temp['tipo_3'] = buildtype[2]

        # Reordenar según ordered_fields
        features = {field: temp.get(field, None) for field in ordered_fields}
        resultados.append(features)

    return resultados, scaler

def crear_grafo_vecindad(resultados):
    """
    Crea un grafo PyTorch Geometric a partir de los resultados procesados de viviendas.
    - Nodos: viviendas (features = todos los atributos excepto id, precio, latitud, longitud, barrio)
    - Aristas: entre viviendas a <= radio_km, o con la más cercana si está aislada
    - Peso de la arista: distancia en km
    """

    radio_km=0.05  # Radio de vecindad en km
    # Extraer coordenadas y features
    coords = []
    features = []
    for v in resultados:
        coords.append((v['latitud'], v['longitud']))
        feat = [v[k] for k in FEATURES_ORDER]
        features.append(feat)
    X = np.array(features, dtype=np.float32)
    num_nodes = len(resultados)
    print(features[0])

    # Crear aristas según el radio
    edge_index = []
    edge_attr = []
    for i in range(num_nodes):
        vecinos = []
        for j in range(num_nodes):
            if i == j:
                continue
            dist = haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            if dist <= radio_km:
                edge_index.append([i, j])
                edge_attr.append([dist])
                vecinos.append(j)
        # Si no tiene vecinos, conectar con el más cercano
        if not vecinos:
            min_dist = float('inf')
            min_j = None
            for j in range(num_nodes):
                if i == j:
                    continue
                dist = haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                if dist < min_dist:
                    min_dist = dist
                    min_j = j
            if min_j is not None:
                edge_index.append([i, min_j])
                edge_attr.append([min_dist])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2,0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr else torch.empty((0,1), dtype=torch.float32)
    x = torch.tensor(X, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    print(f"Creado grafo con {num_nodes} nodos y {edge_index.size(1)} aristas.")
    return data
    
class GCNRegressor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, 1))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr.squeeze()
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return F.relu(x).squeeze()

def predecir_precio_vivienda_con_pesos(
    barrio, data, vivienda_features, vivienda_coord, scaler_precio, scaler_features
):
    """
    Predice el precio de una vivienda usando los hiperparámetros y pesos del modelo asociados al barrio.

    Args:
        barrio (Barriada): Instancia del modelo Barriada.
        data: Grafo Data de PyTorch Geometric.
        vivienda_features: Lista de features de la vivienda.
        vivienda_coord: Coordenadas de la vivienda.
        scaler_precio: Scaler para desnormalizar el precio.
        scaler_features: Scaler para normalizar los features.

    Returns:
        float: Precio predicho.
    """
    hiper = obtener_hiperparametro_barrio(barrio)
    if hiper is None:
        raise ValueError(f"No hay hiperparámetros para el barrio {barrio}")

    # Inicializa el modelo con los hiperparámetros del barrio
    in_channels = data.x.shape[1]
    model = GCNRegressor(
        in_channels,
        hiper.hidden_channels,
        hiper.num_layers,
        hiper.dropout
    )

    # Construye la ruta del archivo de pesos automáticamente
    base_dir = Path(__file__).resolve().parent.parent
    pesos_path = base_dir / "webapp" / "data" / "weight" / f"pesos_gcn_{barrio.id}.pt"
    model.load_state_dict(torch.load(pesos_path, map_location='cpu'))

    # Llama a la función de predicción existente
    return predecir_precio_vivienda(
        model, data, vivienda_features, vivienda_coord, scaler_precio, scaler_features
    )
def predecir_precio_vivienda(model, data, vivienda_features, vivienda_coord, scaler_precio, scaler_features):

    """
    Predice el precio de una vivienda a partir de sus atributos integrándola en el grafo.
    Normaliza los features numéricos de la vivienda con el scaler de procesar_viviendas_barrio.
    El precio predicho se desnormaliza antes de devolverlo.
    """
    radio_km = 0.05

    if isinstance(vivienda_features, dict):
        vivienda_features = [vivienda_features[k] for k in FEATURES_ORDER]

    # Campos numéricos a normalizar (fijos)
    num_fields = [
        'metros_construidos', 'num_hab', 'num_wc', 'planta', 'plantas_edicio_catastro',
        'calidad_catastro', 'distancia_centro', 'distancia_metro', 'distancia_blasco',
        'antiguedad'
    ]

    vivienda_features = np.array(vivienda_features, dtype=np.float32)

    # Normaliza los features numéricos
    feature_keys = [k for k in data.x[0]._fields] if hasattr(data.x[0], '_fields') else None
    if feature_keys:
        for i, k in enumerate(feature_keys):
            if k in num_fields:
                idx = num_fields.index(k)
                vivienda_features[i] = scaler_features.transform([[vivienda_features[j] for j, key in enumerate(feature_keys) if key in num_fields]])[0][idx]
    else:
        # Si no tienes los nombres, asume que los campos numéricos están en las mismas posiciones que en el grafo
        # y que el orden de vivienda_features es el mismo que en crear_grafo_vecindad
        # Busca los índices de los campos numéricos en el vector de features
        # Por ejemplo, si sabes que los campos numéricos están en las posiciones [0,1,2,3,4,5,6,7,8,9]
        # puedes hacer:
        idxs_num_fields = [i for i, k in enumerate(num_fields) if k in num_fields]
        num_values = [vivienda_features[i] for i in idxs_num_fields]
        norm_values = scaler_features.transform([num_values])[0]
        for idx, norm_val in zip(idxs_num_fields, norm_values):
            vivienda_features[idx] = norm_val

    x_new = torch.cat([data.x, torch.tensor([vivienda_features], dtype=torch.float32)], dim=0)
    coords = [tuple(c) for c in data.x.cpu().numpy()]
    coords.append(vivienda_coord)
    num_nodes = x_new.shape[0]

    edge_index = data.edge_index.cpu().numpy().tolist()
    edge_attr = data.edge_attr.cpu().numpy().tolist()

    vecinos = []
    for j in range(num_nodes - 1):
        dist = haversine(vivienda_coord[0], vivienda_coord[1], coords[j][0], coords[j][1])
        if dist <= radio_km:
            edge_index[0].append(num_nodes - 1)
            edge_index[1].append(j)
            edge_attr.append([dist])
            vecinos.append(j)
    if not vecinos:
        min_dist = float('inf')
        min_j = None
        for j in range(num_nodes - 1):
            dist = haversine(vivienda_coord[0], vivienda_coord[1], coords[j][0], coords[j][1])
            if dist < min_dist:
                min_dist = dist
                min_j = j
        if min_j is not None:
            edge_index[0].append(num_nodes - 1)
            edge_index[1].append(min_j)
            edge_attr.append([min_dist])

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    data_new = Data(x=x_new, edge_index=edge_index, edge_attr=edge_attr)

    # Predecir y desnormalizar el precio
    model.eval()
    with torch.no_grad():
        pred_norm = model(data_new)[-1].cpu().numpy().reshape(1, -1)
        pred_real = scaler_precio.inverse_transform(pred_norm)[0, 0]
        pred_real = float(np.clip(pred_real, 480, 9500))
    return pred_real

def obtener_hiperparametro_barrio(barrio):
    """
    Devuelve la instancia del modelo Hiperparametro asociada a un barrio dado.

    Parametros:
        barrio (Barriada): Instancia del modelo Barriada.

    Returns:
        Hiperparametro: Instancia asociada al barrio, o None si no existe.
    """
    try:
        return barrio.hiperparametros.first()
    except Exception as e:
        print(f"Error al obtener hiperparámetro para el barrio {barrio}: {e}")
        return None

def generar_id_vivienda_unico():
    """
    Genera un id único para una vivienda, formado por la letra 'A' y una secuencia de números.
    Ejemplo: 'A18333312729820435049'
    Comprueba que no exista ya en la base de datos.
    """
    while True:
        letra = 'A'
        numeros = ''.join(random.choices(string.digits, k=17))
        nuevo_id = f"{letra}{numeros}"
        if not Vivienda.objects.filter(id=nuevo_id).exists():
            return nuevo_id
        
