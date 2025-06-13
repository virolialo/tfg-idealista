from math import radians, sin, cos, sqrt, atan2
from .models import Metro, Barriada, Vivienda
import json
from pathlib import Path
from shapely.geometry import shape, Point
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import optuna
from pathlib import Path
import torch
from torch_geometric.data import DataLoader
from django.conf import settings
from .models import Hiperparametro
import random
import string
import csv
from .models import Hiperparametro


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

def procesar_viviendas_barrio(barrio):
    """
    Dado un barrio, obtiene todas las Viviendas asociadas y:
    1. Transforma los atributos booleanos a 0/1.
    2. Codifica el atributo 'estado' en tres columnas tipo_1, tipo_2, tipo_3.
    3. Ordena los atributos según el orden especificado.
    4. Normaliza el target (precio_m2) y los features numéricos seleccionados.

    Returns:
        viviendas_procesadas: lista de diccionarios con los atributos ordenados y transformados.
        scaler_features: scaler ajustado a los features numéricos.
        scaler_target: scaler ajustado al target.
    """

    # Si barrio es una instancia, obtener su id
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

    viviendas_lista = []
    for v in viviendas:
        d = {}
        # Atributos básicos
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

        # Booleanos a 0/1
        for b in booleanos:
            d[b] = 1 if getattr(v, b) else 0

        # Codificación del estado
        if v.estado == "NEWCONSTRUCTION":
            d['tipo_1'], d['tipo_2'], d['tipo_3'] = 1, 0, 0
        elif v.estado == "2HANDRESTORE":
            d['tipo_1'], d['tipo_2'], d['tipo_3'] = 0, 1, 0
        else:  # "2HANDGOOD"
            d['tipo_1'], d['tipo_2'], d['tipo_3'] = 0, 0, 1

        viviendas_lista.append(d)

    # Normalización
    # Target
    precios = np.array([v['precio_m2'] for v in viviendas_lista]).reshape(-1, 1)
    scaler_target = MinMaxScaler()
    precios_norm = scaler_target.fit_transform(precios)
    for i, v in enumerate(viviendas_lista):
        v['precio_m2'] = precios_norm[i, 0]

    # Features
    X_features = np.array([[v[f] for f in features_a_normalizar] for v in viviendas_lista])
    scaler_features = MinMaxScaler()
    X_features_norm = scaler_features.fit_transform(X_features)
    for i, v in enumerate(viviendas_lista):
        for j, f in enumerate(features_a_normalizar):
            v[f] = X_features_norm[i, j]

    # Orden final (sin id ni barrio)
    viviendas_procesadas = []
    for v in viviendas_lista:
        v_ordenado = {k: v[k] for k in orden}
        viviendas_procesadas.append(v_ordenado)
    print(f"Procesadas {len(viviendas_procesadas)} viviendas del barrio {barrio_id}.")
    return viviendas_procesadas, scaler_features, scaler_target

def crear_grafo_vecindad(viviendas_procesadas):
    """
    Crea un grafo PyTorch Geometric a partir de una lista de viviendas procesadas.
    - Crea aristas entre viviendas a menos de radio_km (por defecto 0.05 km) usando la distancia como peso.
    - Si un nodo queda aislado, se conecta a su vecino más cercano.
    - Elimina los atributos 'id', 'precio_m2', 'barrio', 'latitud' y 'longitud' de los features de cada nodo.

    Args:
        viviendas_procesadas (list[dict]): Lista de viviendas procesadas.
        radio_km (float): Radio de vecindad en kilómetros.

    Returns:
        Data: Grafo PyTorch Geometric.
    """
    n = len(viviendas_procesadas)
    latitudes = np.array([v['latitud'] for v in viviendas_procesadas])
    longitudes = np.array([v['longitud'] for v in viviendas_procesadas])

    radio_km = 0.05

    # Construir matriz de distancias
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = haversine(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Crear aristas por vecindad
    edge_index = []
    edge_attr = []
    for i in range(n):
        vecinos = []
        for j in range(n):
            if i != j and dist_matrix[i, j] <= radio_km:
                edge_index.append([i, j])
                edge_attr.append([dist_matrix[i, j]])
                vecinos.append(j)
        # Si no tiene vecinos, conectar con el más cercano
        if not vecinos:
            nearest = np.argmin(dist_matrix[i] + np.eye(n)[i]*1e6)
            edge_index.append([i, nearest])
            edge_attr.append([dist_matrix[i, nearest]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Features de los nodos (eliminando 'id', 'precio_m2', 'barrio', 'latitud', 'longitud')
    features_keys = [k for k in viviendas_procesadas[0].keys() if k not in ('precio_m2', 'latitud', 'longitud')]
    x = torch.tensor([[v[k] for k in features_keys] for v in viviendas_procesadas], dtype=torch.float)

    # Target (precio_m2 normalizado)
    y = torch.tensor([v['precio_m2'] for v in viviendas_procesadas], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    print(f"Creado grafo con {data.num_nodes} nodos y {data.num_edges} aristas.")

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

def entrenar_gcn_optuna(grafo, barrio_id, scaler_target=None):
    """
    Entrena el modelo GCN usando Optuna para optimizar hiperparámetros.
    Divide los nodos en train/test, entrena solo con train y evalúa en test.
    Guarda los mejores hiperparámetros en la BD y los pesos en disco.
    Devuelve métricas de evaluación, incluyendo errores desnormalizados si se pasa scaler_target.
    """
    # Valores fijos
    n_trials = 20
    max_epochs = 200
    test_ratio = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = grafo.to(device)
    n_nodes = data.num_nodes

    # División aleatoria de nodos en train/test
    idx = np.arange(n_nodes)
    np.random.shuffle(idx)
    split = int(n_nodes * (1 - test_ratio))
    train_idx = torch.tensor(idx[:split], dtype=torch.long, device=device)
    test_idx = torch.tensor(idx[split:], dtype=torch.long, device=device)

    def objective(trial):
        hidden_channels = trial.suggest_int("hidden_channels", 8, 128)
        num_layers = trial.suggest_int("num_layers", 2, 5)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        epochs = trial.suggest_int("epochs", 50, max_epochs)

        model = GCNRegressor(
            in_channels=data.x.shape[1],
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data)[train_idx]
            loss = loss_fn(out, data.y[train_idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(data)[test_idx]
            target = data.y[test_idx]
            mse = torch.mean((pred - target) ** 2).item()
            mae = torch.mean(torch.abs(pred - target)).item()
            ss_res = torch.sum((pred - target) ** 2).item()
            ss_tot = torch.sum((target - torch.mean(target)) ** 2).item()
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')
        trial.set_user_attr("mae", mae)
        trial.set_user_attr("epochs", epochs)
        trial.set_user_attr("r2", r2)
        return mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_mae = study.best_trial.user_attrs["mae"]
    best_epochs = study.best_trial.user_attrs["epochs"]
    best_r2 = study.best_trial.user_attrs["r2"]

    # Entrena el modelo final con los mejores hiperparámetros
    best_model = GCNRegressor(
        in_channels=data.x.shape[1],
        hidden_channels=best_params["hidden_channels"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"]
    ).to(device)
    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params["lr"])
    loss_fn = torch.nn.MSELoss()
    best_model.train()
    for epoch in range(best_epochs):
        optimizer.zero_grad()
        out = best_model(data)[train_idx]
        loss = loss_fn(out, data.y[train_idx])
        loss.backward()
        optimizer.step()

    # Guarda los hiperparámetros en la BD
    Hiperparametro.objects.update_or_create(
        barrio_id=barrio_id,
        defaults={
            "hidden_channels": best_params["hidden_channels"],
            "num_layers": best_params["num_layers"],
            "dropout": best_params["dropout"],
            "lr": best_params["lr"],
            "epochs": best_epochs,
        }
    )

    # Guarda los pesos
    BASE_DIR = Path(settings.BASE_DIR)
    weight_dir = BASE_DIR / "webapp" / "data" / "weight"
    weight_dir.mkdir(parents=True, exist_ok=True)
    weight_path = weight_dir / f"pesos_gcn_{barrio_id}.pt"
    torch.save(best_model.state_dict(), weight_path)

    # Métricas finales en test
    best_model.eval()
    with torch.no_grad():
        pred = best_model(data)[test_idx].cpu().numpy()
        target = data.y[test_idx].cpu().numpy()
        mse = np.mean((pred - target) ** 2)
        mae = np.mean(np.abs(pred - target))
        ss_res = np.sum((pred - target) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')

        # Desnormalización si se pasa scaler_target
        if scaler_target is not None:
            pred_desnorm = scaler_target.inverse_transform(pred.reshape(-1, 1)).flatten()
            target_desnorm = scaler_target.inverse_transform(target.reshape(-1, 1)).flatten()
            mse_desnorm = np.mean((pred_desnorm - target_desnorm) ** 2)
            mae_desnorm = np.mean(np.abs(pred_desnorm - target_desnorm))
        else:
            mse_desnorm = None
            mae_desnorm = None

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "mse_desnorm": mse_desnorm,
        "mae_desnorm": mae_desnorm,
        "best_params": best_params,
        "epochs": best_epochs,
        "weight_path": str(weight_path)
    }

def generar_id_vivienda_unico():
    """
    Genera un ID aleatorio para vivienda que empieza por 'A' seguido de dígitos,
    y comprueba que no exista ya en la base de datos. Si existe, repite hasta encontrar uno único.
    Ejemplo de ID: A9651535568269959084
    """
    while True:
        # Genera una cadena: 'A' + 18 dígitos aleatorios
        id_candidato = 'A' + ''.join(random.choices(string.digits, k=18))
        if not Vivienda.objects.filter(id=id_candidato).exists():
            return id_candidato
        
def procesar_nueva_vivienda_formulario(
    datos_formulario,
    distancia_blasco,
    distancia_metro,
    distancia_centro,
    scaler_features
):
    """
    Procesa y normaliza los datos de una nueva vivienda recibidos desde un formulario,
    usando el mismo procesamiento y normalización que en procesar_viviendas_barrio(),
    pero sin incluir precio_m2.

    Args:
        datos_formulario (dict): Diccionario con los datos del formulario.
        distancia_blasco (float): Distancia a Blasco Ibáñez (km)
        distancia_metro (float): Distancia a metro más cercano (km)
        distancia_centro (float): Distancia al centro (km)
        scaler_features (MinMaxScaler): Scaler ajustado a los features numéricos.

    Returns:
        dict: Diccionario con los datos procesados y normalizados, en el orden especificado.
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
        'tipo_1', 'tipo_2', 'tipo_3', 'antiguedad', 'precio_m2'  # Añadimos precio_m2 al final
    ]

    v = {}
    # Atributos básicos
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

    # Booleanos a 0/1
    for b in booleanos:
        v[b] = 1 if datos_formulario.get(b, False) else 0

    # Codificación del estado
    estado = datos_formulario.get('estado', '2HANDGOOD')
    if estado == "NEWCONSTRUCTION":
        v['tipo_1'], v['tipo_2'], v['tipo_3'] = 1, 0, 0
    elif estado == "2HANDRESTORE":
        v['tipo_1'], v['tipo_2'], v['tipo_3'] = 0, 1, 0
    else:  # "2HANDGOOD" u otro
        v['tipo_1'], v['tipo_2'], v['tipo_3'] = 0, 0, 1

    # Normalización de features numéricos
    X_features = np.array([[v[f] for f in features_a_normalizar]])
    X_features_norm = scaler_features.transform(X_features)
    for j, f in enumerate(features_a_normalizar):
        v[f] = X_features_norm[0, j]

    # Añadir precio_m2 ficticio para predicción
    v['precio_m2'] = 0

    # Orden final
    v_ordenado = {k: v.get(k, None) for k in orden}
    return v_ordenado

def predecir_precio_m2_vivienda(
    viviendas_procesadas,
    vivienda_proc,
    barrio_id,
    scaler_target
):
    """
    Integra la nueva vivienda en el grafo de vecindad, predice su precio_m2 usando el modelo GCN entrenado
    y devuelve el precio desnormalizado.

    Args:
        viviendas_procesadas (list[dict]): Lista de viviendas procesadas del barrio.
        vivienda_proc (dict): Diccionario procesado y normalizado de la nueva vivienda.
        barrio_id (int): ID del barrio.
        scaler_target (MinMaxScaler): Scaler ajustado al target (precio_m2).

    Returns:
        float: Predicción desnormalizada de precio_m2 para la nueva vivienda.
    """
    import torch
    from pathlib import Path
    from django.conf import settings
    from .models import Hiperparametro

    # 1. Añadir la nueva vivienda a la lista de viviendas procesadas
    viviendas_full = viviendas_procesadas + [vivienda_proc]

    # 2. Crear el nuevo grafo con la vivienda añadida
    grafo_nuevo = crear_grafo_vecindad(viviendas_full)

    # 3. Cargar hiperparámetros y pesos del modelo
    params = Hiperparametro.objects.get(barrio_id=barrio_id)
    BASE_DIR = Path(settings.BASE_DIR)
    weight_path = BASE_DIR / "webapp" / "data" / "weight" / f"pesos_gcn_{barrio_id}.pt"
    print(f"Cargando pesos del modelo desde: {weight_path}")
    in_channels = grafo_nuevo.x.shape[1]
    model = GCNRegressor(
        in_channels=in_channels,
        hidden_channels=params.hidden_channels,
        num_layers=params.num_layers,
        dropout=params.dropout
    )
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()

    # 4. Predecir para el último nodo (la nueva vivienda)
    with torch.no_grad():
        out = model(grafo_nuevo)
        pred_norm = out[-1].item()  # Último nodo es la nueva vivienda

    # 5. Desnormalizar el resultado
    precio_predicho = scaler_target.inverse_transform([[pred_norm]])[0, 0]

    # 6. Limitar el precio_m2 a un rango razonable
    precio_predicho = max(500, min(precio_predicho, 4000))
    print(f"Precio predicho (desnormalizado): {precio_predicho}")
    return precio_predicho

def obtener_vecinos_vivienda(barrio_id, vivienda_proc):
    """
    Dado un id de barrio y una vivienda procesada (diccionario con los mismos campos que las viviendas del barrio),
    crea el grafo de vecindad y devuelve la lista de viviendas del barrio que están conectadas (vecinas) a la vivienda pasada.

    Args:
        barrio_id (int): ID del barrio.
        vivienda_proc (dict): Diccionario procesado de la vivienda (con latitud y longitud).

    Returns:
        list[dict]: Lista de viviendas vecinas (diccionarios) del barrio, solo con latitud y longitud.
    """
    radio_km = 0.05  # Fijo
    # Obtener viviendas procesadas del barrio
    viviendas_procesadas, _, _ = procesar_viviendas_barrio(barrio_id)
    # Añadir la vivienda nueva al final
    viviendas_full = viviendas_procesadas + [vivienda_proc]

    # Crear grafo de vecindad
    n = len(viviendas_full)
    latitudes = np.array([v['latitud'] for v in viviendas_full])
    longitudes = np.array([v['longitud'] for v in viviendas_full])

    # Construir matriz de distancias
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = haversine(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Buscar vecinos de la última vivienda (la pasada por parámetro)
    idx_vivienda = n - 1
    vecinos_idx = [i for i in range(n-1) if dist_matrix[idx_vivienda, i] <= radio_km]

    # Si no tiene vecinos, conectar con el más cercano
    if not vecinos_idx and n > 1:
        nearest = np.argmin(dist_matrix[idx_vivienda, :-1])
        vecinos_idx = [nearest]

    # Devolver solo latitud y longitud
    vecinos = []
    for v in [viviendas_procesadas[i] for i in vecinos_idx]:
        vecinos.append({
            'latitud': float(v['latitud']),
            'longitud': float(v['longitud'])
        })
    return vecinos

def exportar_hiperparametros_csv(ruta_csv):
    """
    Exporta todos los hiperparámetros registrados en la BD a un archivo CSV.
    Los campos flotantes se exportan con todos sus decimales.
    Cabecera: ID,HIDDENCHANNELS,LAYERS,DROPOUT,LR,EPOCHS
    """
    hiperparams = Hiperparametro.objects.all()
    with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'HIDDENCHANNELS', 'LAYERS', 'DROPOUT', 'LR', 'EPOCHS'])
        for h in hiperparams:
            writer.writerow([
                h.barrio_id,
                h.hidden_channels,
                h.num_layers,
                repr(h.dropout),  # todos los decimales
                repr(h.lr),       # todos los decimales
                h.epochs
            ])