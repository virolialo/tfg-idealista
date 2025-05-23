import pandas as pd
from pathlib import Path
import itertools
import numpy as np
from sklearn.neighbors import NearestNeighbors, BallTree
import torch
from torch_geometric.data import Data

def exportar_nodos(csv_path, ruta):
    """
    Exporta un archivo CSV de viviendas a nodos.

    El archivo se guarda en graphs/Valencia_nodes.csv

    Parámetros:
    csv_path (str): Ruta al archivo CSV de entrada.
    ruta (str): Ruta donde se guardará el archivo CSV de salida.

    Returns:
    Ninguno
    """
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
    Crea un archivo CSV con las aristas en funcion de si
    dos nodos pertenecen al mismo barrio o no.

    El archivo se guarda en graph/Valencia_neighbour_edges.csv

    Parámetros:
    csv_path (str): Ruta al archivo CSV de entrada.
    ruta (str): Ruta donde se guardará el archivo CSV de salida.

    Returns:
    Ninguno
    """
    df = pd.read_csv(csv_path)
    edge_rows = []
    for _, group in df.groupby("NEIGHBOURID"):
        idx = group.index.to_numpy()
        if len(idx) > 1:
            comb = np.array(list(itertools.combinations(idx, 2)))
            edge_rows.extend([{"source": i, "target": j} for i, j in comb])
    df_edges = pd.DataFrame(edge_rows)
    edges_out = Path(ruta) / "Valencia_neighbour_edges.csv"
    df_edges.to_csv(edges_out, index=False, encoding='utf-8')
    print(f"Aristas exportadas a {edges_out}")

def exportar_aristas_vecindad(csv_path, ruta, radio_km):
    """
    Se crea un archivo CSV con las aristas en funcion de si
    dos nodos se encuentran dentro del mismo radio, considerando
    asi que son vecinos.

    El archivo se guarda en graph/Valencia_kdd_edges_<radio>.csv

    Parametros:
    csv_path (str): Ruta al archivo CSV de entrada.
    ruta (str): Ruta donde se guardara el archivo CSV de salida.
    radio_km (float): Radio en km para considerar dos nodos como vecinos.

    Returns:
    Ninguno
    """
    df = pd.read_csv(csv_path)
    coords = df[["LATITUDE", "LONGITUDE"]].values
    coords_rad = np.radians(coords)
    tree = BallTree(coords_rad, metric='haversine')
    radio_rad = radio_km / 6371.0

    # Busqueda de vecinos
    ind_array, dist_array = tree.query_radius(coords_rad, r=radio_rad, return_distance=True)
    edge_rows = set()
    for i, (neighbors, distances) in enumerate(zip(ind_array, dist_array)):
        for j, dist in zip(neighbors, distances):
            if i < j:
                edge_rows.add((i, j, dist * 6371.0))
                print(f"Arista: {i} - {j} (distancia: {dist * 6371.0:.4f} km)")

    df_edges = pd.DataFrame(list(edge_rows), columns=["source", "target", "DISTANCE"])
    radio_str = f"{radio_km:.3f}".replace('.', '')
    edges_out = Path(ruta) / f"Valencia_kdd_edges_{radio_str}.csv"
    df_edges.to_csv(edges_out, index=False, encoding='utf-8')
    print(f"Aristas por vecindad exportadas a {edges_out}")

def exportar_aristas_similitud_caracteristicas(csv_path, ruta, threshold):
    """
    La funcion crea un archivo CSV con las aristas en funcion de la
    similitud de las caracteristicas de los nodos. Esto se establece
    mediante la distancia euclidea entre los nodos, y se considera
    que dos nodos son similares si la distancia es menor que un
    umbral determinado. 

    El archivo se guarda en graph/Valencia_similarity_edges_<umbral>.csv

    Parametros:
    csv_path (str): Ruta al archivo CSV de entrada.
    ruta (str): Ruta donde se guardará el archivo CSV de salida.
    threshold (float): Umbral de similitud (0 < umbral < 1).

    Returns:
    Ninguno
    """
    df = pd.read_csv(csv_path)
    atributos = [
        "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER", "HASTERRACE", "HASLIFT",
        "HASAIRCONDITIONING", "HASPARKINGSPACE", "HASNORTHORIENTATION", "HASSOUTHORIENTATION",
        "HASEASTORIENTATION", "HASWESTORIENTATION", "HASBOXROOM", "HASWARDROBE",
        "HASSWIMMINGPOOL", "HASDOORMAN", "HASGARDEN", "ISDUPLEX", "ISSTUDIO",
        "ISINTOPFLOOR", "FLOORCLEAN"
    ]
    X = df[atributos].values
    n_features = len(atributos)
    dist_threshold = 1 - threshold
    dist_threshold = dist_threshold * np.sqrt(n_features)

    nn = NearestNeighbors(radius=dist_threshold, metric='euclidean', n_jobs=-1)
    nn.fit(X)
    ind_array = nn.radius_neighbors(X, return_distance=False)

    edge_rows = set()
    for i, neighbors in enumerate(ind_array):
        for j in neighbors:
            if i < j:
                dist = np.linalg.norm(X[i] - X[j]) / np.sqrt(n_features)
                similarity = 1 - dist
                edge_rows.add((i, j, similarity))
                print(f"Arista: {i} - {j} (similitud: {similarity:.4f})")

    df_edges = pd.DataFrame(list(edge_rows), columns=["source", "target", "SIMILARITY"])
    threshold_str = f"{threshold:.3f}".replace('.', '')
    edges_out = Path(ruta) / f"Valencia_similarity_edges_{threshold_str}.csv"
    df_edges.to_csv(edges_out, index=False, encoding='utf-8')
    print(f"Aristas por similitud exportadas a {edges_out}")

def cargar_grafo(nodes_csv, edges_csv):
    """
    Crea un grafo con PyTorch Geometric a partir de los archivos CSV de nodos y aristas.

    Parametros:
    nodes_csv (str): Ruta al archivo CSV de nodos.
    edges_csv (str): Ruta al archivo CSV de aristas.

    Returns:
    data (torch_geometric.data.Data): Objeto Data que representa el grafo.
    """
    nodes_df = pd.read_csv(nodes_csv) # Nodos
    edges_df = pd.read_csv(edges_csv) # Aristas

    # Carateristicas de los nodos (no incluir NODEID por ser el identificador)
    node_features = nodes_df.drop(columns=["NODEID"]).values
    x = torch.tensor(node_features, dtype=torch.float)

    edge_index = torch.tensor(edges_df[["source", "target"]].values.T, dtype=torch.long)

    # Si existe peso en el CSV de aristas
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

