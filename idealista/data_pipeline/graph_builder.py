import pandas as pd
from pathlib import Path
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors, BallTree
import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

def exportar_nodos(csv_path, ruta):
    node_cols = [
        "PRICE", "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER",
        "HASTERRACE", "HASLIFT", "HASAIRCONDITIONING", "HASPARKINGSPACE",
        "HASNORTHORIENTATION", "HASSOUTHORIENTATION", "HASEASTORIENTATION", "HASWESTORIENTATION",
        "HASBOXROOM", "HASWARDROBE", "HASSWIMMINGPOOL", "HASDOORMAN", "HASGARDEN",
        "ISDUPLEX", "ISSTUDIO", "ISINTOPFLOOR", "FLOORCLEAN", "CADMAXBUILDINGFLOOR",
        "CADASTRALQUALITYID", "BUILTTYPEID_1", "BUILTTYPEID_2", "BUILTTYPEID_3",
        "DISTANCE_TO_CITY_CENTER", "DISTANCE_TO_METRO", "DISTANCE_TO_BLASCO",
        "LONGITUDE", "LATITUDE", "ANTIQUITY", "NEIGHBOURID"
    ]
    df = pd.read_csv(csv_path)
    df_nodes = df[node_cols].copy()
    df_nodes.insert(0, "NODEID", df_nodes.index)
    nodes_out = Path(ruta) / "Valencia_nodes.csv"
    df_nodes.to_csv(nodes_out, index=False, encoding='utf-8')
    print(f"Nodos exportados a {nodes_out}")

def exportar_aristas_barrio(csv_path, ruta):
    """
    Exporta un archivo CSV con las aristas:
    - Dos viviendas están conectadas si pertenecen al mismo barrio (NEIGHBOURID)
    El archivo se guarda en ruta/Valencia_neighbour_edges.csv
    """
    df = pd.read_csv(csv_path)
    edge_rows = []
    for _, group in df.groupby("NEIGHBOURID"):
        idx = group.index.to_numpy()
        if len(idx) > 1:
            # Solo una arista por par (i < j) para grafo no dirigido
            comb = np.array(list(itertools.combinations(idx, 2)))
            edge_rows.extend([{"source": i, "target": j} for i, j in comb])
    df_edges = pd.DataFrame(edge_rows)
    edges_out = Path(ruta) / "Valencia_neighbour_edges.csv"
    df_edges.to_csv(edges_out, index=False, encoding='utf-8')
    print(f"Aristas exportadas a {edges_out}")

def exportar_aristas_vecindad(csv_path, ruta, radio_km):
    """
    Exporta un archivo CSV con las aristas:
    - Solo por vecindad geográfica (dentro de radio_km)
    Cada arista tiene como peso la distancia (en km) entre las viviendas.
    El archivo se guarda en ruta/Valencia_neighbour_edges_by_distance.csv
    Optimizado con BallTree (sklearn).
    """
    df = pd.read_csv(csv_path)
    coords = df[["LATITUDE", "LONGITUDE"]].values
    coords_rad = np.radians(coords)
    tree = BallTree(coords_rad, metric='haversine')
    radio_rad = radio_km / 6371.0  # radio en radianes

    # Buscar vecinos para cada nodo y calcular distancias
    ind_array, dist_array = tree.query_radius(coords_rad, r=radio_rad, return_distance=True)
    edge_rows = set()
    for i, (neighbors, distances) in enumerate(zip(ind_array, dist_array)):
        for j, dist in zip(neighbors, distances):
            if i < j:  # Solo una arista por par para grafo no dirigido
                edge_rows.add((i, j, dist * 6371.0))  # dist está en radianes, convertir a km
                print(f"Arista: {i} - {j} (distancia: {dist * 6371.0:.4f} km)")

    # Exportar a CSV
    df_edges = pd.DataFrame(list(edge_rows), columns=["source", "target", "DISTANCE"])
    radio_str = f"{radio_km:.3f}".replace('.', '')
    edges_out = Path(ruta) / f"Valencia_kdd_edges_{radio_str}.csv"
    df_edges.to_csv(edges_out, index=False, encoding='utf-8')
    print(f"Aristas por vecindad exportadas a {edges_out}")

def exportar_aristas_similitud_caracteristicas(csv_path, ruta, threshold):
    """
    Exporta un archivo CSV con las aristas:
    - Dos viviendas están conectadas si su similitud (1 - distancia euclídea media)
      sobre los atributos seleccionados es mayor o igual al umbral (threshold).
    El archivo se guarda en ruta/Valencia_similarity_edges.csv
    """
    df = pd.read_csv(csv_path)
    atributos = [
        "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER", "HASTERRACE", "HASLIFT",
        "HASAIRCONDITIONING", "HASPARKINGSPACE", "HASNORTHORIENTATION", "HASSOUTHORIENTATION",
        "HASEASTORIENTATION", "HASWESTORIENTATION", "HASBOXROOM", "HASWARDROBE",
        "HASSWIMMINGPOOL", "HASDOORMAN", "HASGARDEN", "ISDUPLEX", "ISSTUDIO",
        "ISINTOPFLOOR", "FLOORCLEAN"
    ]
    X = df[atributos].values  # Ya normalizados
    n_features = len(atributos)
    dist_threshold = 1 - threshold
    dist_threshold = dist_threshold * np.sqrt(n_features)

    nn = NearestNeighbors(radius=dist_threshold, metric='euclidean', n_jobs=-1)
    nn.fit(X)
    ind_array = nn.radius_neighbors(X, return_distance=False)

    edge_rows = set()
    for i, neighbors in enumerate(ind_array):
        for j in neighbors:
            if i < j:  # Solo una arista por par para grafo no dirigido
                dist = np.linalg.norm(X[i] - X[j]) / np.sqrt(n_features)
                similarity = 1 - dist
                edge_rows.add((i, j, similarity))
                print(f"Arista: {i} - {j} (similitud: {similarity:.4f})")

    df_edges = pd.DataFrame(list(edge_rows), columns=["source", "target", "SIMILARITY"])
    threshold_str = f"{threshold:.3f}".replace('.', '')
    edges_out = Path(ruta) / f"Valencia_similarity_edges_{threshold_str}.csv"
    df_edges.to_csv(edges_out, index=False, encoding='utf-8')
    print(f"Aristas por similitud exportadas a {edges_out}")

def cargar_grafo(nodes_csv, edges_csv,max_nodos=100, max_aristas=200):
    # Leer nodos y aristas
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    # Extraer características de los nodos (excluyendo NODEID)
    node_features = nodes_df.drop(columns=["NODEID"]).values
    x = torch.tensor(node_features, dtype=torch.float)

    # Construir edge_index
    edge_index = torch.tensor(edges_df[["source", "target"]].values.T, dtype=torch.long)

    # Si hay una tercera columna, usarla como edge_attr (peso)
    edge_attr = None
    if edges_df.shape[1] > 2:
        weight_col = edges_df.columns[2]
        edge_attr = torch.tensor(edges_df[weight_col].values, dtype=torch.float).unsqueeze(1)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    else:
        data = Data(x=x, edge_index=edge_index)

    return data

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    csv_in = BASE_DIR / "processed_data" / "Valencia_Sale_graph.csv"
    ruta = BASE_DIR / "graphs"
#    exportar_nodos(csv_in, ruta)
    nodos = BASE_DIR / "graphs" / "Valencia_nodes.csv"
    aristas_1 = BASE_DIR / "graphs" / "Valencia_kdd_edges_0050.csv"
#    exportar_aristas_barrio(nodos, ruta)
#    exportar_aristas_vecindad(nodos, ruta, 0.3)
#    exportar_aristas_similitud_caracteristicas(nodos, ruta, 0.99)
    cargar_grafo(nodos, aristas_1, max_nodos=2000, max_aristas=20000)
