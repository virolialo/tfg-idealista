import pandas as pd
from pathlib import Path
import itertools
import numpy as np
from sklearn.neighbors import NearestNeighbors, BallTree
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx

def exportar_nodos(csv_entrada):
    """
    La funcion crea archivos CSV para definir los nodos de un grafo a partir
    de un archivo CSV de viviendas. Pueden generarse dos tipos de archivos:
    - Un archivo por barrio, que contiene los nodos de un barrio concreto.
    - Un archivo con todos los nodos, que contiene todas la viviendas de Valencia.

    Los archivos se guardan en graphs/nodes/Valencia_nodes_{NEIGHBOURID}.csv
    y graphs/nodes/Valencia_nodes_all.csv, respectivamente.

    Parámetros:
    csv_entrada (str): Ruta al archivo CSV de entrada.

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

    df = pd.read_csv(csv_entrada) # Viviendas
    barrios = df["NEIGHBOURID"].unique()

    # Directorio de salida
    BASE_DIR = Path(__file__).resolve().parent
    nodes_dir = BASE_DIR / "graphs" / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)

    # Archivo CSV por barrio
    for barrio in barrios:
        df_barrio = df[df["NEIGHBOURID"] == barrio][node_cols].copy()
        df_barrio.insert(0, "NODEID", df_barrio.index)
        nodes_out = nodes_dir / f"Valencia_nodes_{barrio}.csv"
        df_barrio.to_csv(nodes_out, index=False, encoding='utf-8')
        print(f"Nodos del barrio {barrio} exportados a {nodes_out}")

    # Archivo CSV completo
#    df_all = df[node_cols].copy()
#    df_all.insert(0, "NODEID", df_all.index)
#    nodes_all_out = nodes_dir / "Valencia_nodes_all.csv"
#    df_all.to_csv(nodes_all_out, index=False, encoding='utf-8')
#    print(f"Todos los nodos exportados a {nodes_all_out}")

def exportar_aristas_barrio(nodos):
    """
    La funcion crea un archivo CSV con las aristas de un grafo a partir
    de un archivo CSV de viviendas. Las aristas se crean entre nodos
    que pertenecen al mismo barrio, considerando que son vecinos.

    El archivo se guarda en graphs/edges/neighbour/Valencia_neighbour_edges_<NEIGHBOURID>.csv
    o graphs/edges/neighbour/Valencia_neighbour_edges_all.csv si participan todos los barrios.

    Parametros:
    nodos (str): Ruta al archivo CSV de entrada.

    Returns:
    Ninguno
    """
    # Lectura del CSV
    df = pd.read_csv(nodos)
    edge_rows = []
    barrios = df["NEIGHBOURID"].unique()

    # Agrupar por barrio
    for _, group in df.groupby("NEIGHBOURID"):
        idx = group.index.to_numpy()
        if len(idx) > 1:
            comb = np.array(list(itertools.combinations(idx, 2)))
            edge_rows.extend([{"source": i, "target": j} for i, j in comb])
    df_edges = pd.DataFrame(edge_rows)

    # Directorio de salida
    BASE_DIR = Path(__file__).resolve().parent
    edges_dir = BASE_DIR / "graphs" / "edges" / "neighbour"
    edges_dir.mkdir(parents=True, exist_ok=True)

    # Formato de ficheros de salida
    if len(barrios) == 1:
        edges_out = edges_dir / f"Valencia_neighbour_edges_{barrios[0]}.csv"
    else:
        edges_out = edges_dir / "Valencia_neighbour_edges_all.csv"
    df_edges.to_csv(edges_out, index=False, encoding='utf-8')
    print(f"Aristas exportadas a {edges_out}")

def exportar_aristas_vecindad(nodos, radio_km):
    """
    La funcion crea un archivo CSV con las aristas de un grafo a partir
    de un archivo CSV de viviendas. Las aristas se crean entre nodos
    que se encuentran dentro de un radio determinado, considerando que son vecinos.

    Ademas, si un nodo no tiene vecinos dentro del radio, se conecta al nodo mas cercano,
    asegurando asi que no existen nodos aisalados.

    Parametros:
    nodos (str): Ruta al archivo CSV de entrada.
    radio_km (float): Radio en kilómetros para considerar vecinos.
    
    Returns:
    Ninguno
    """
    # Lectura del CSV
    df = pd.read_csv(nodos)

    # Coordenadas
    coords = df[["LATITUDE", "LONGITUDE"]].values
    coords_rad = np.radians(coords)
    tree = BallTree(coords_rad, metric='haversine')
    radio_rad = radio_km / 6371.0

    # Busqueda de vecinos
    ind_array, dist_array = tree.query_radius(coords_rad, r=radio_rad, return_distance=True)
    edge_rows = set()
    n = len(coords)
    for i, (neighbors, distances) in enumerate(zip(ind_array, dist_array)):
        for j, dist in zip(neighbors, distances):
            if i < j:
                edge_rows.add((i, j, dist * 6371.0))

    # Detectar nodos aislados (sin vecinos)
    vecinos = {i for edge in edge_rows for i in edge[:2]}
    aislados = set(range(n)) - vecinos

    # Busqueda del vecino mas cercano para nodos aislados
    if aislados:
        dists, inds = tree.query(coords_rad, k=n)
        for i in aislados:
            for j, dist in zip(inds[i], dists[i]):
                if i != j:
                    edge_rows.add((min(i, j), max(i, j), dist * 6371.0))
                    break

    df_edges = pd.DataFrame(list(edge_rows), columns=["source", "target", "DISTANCE"])

    # Formateo del radio para nombres de archivo
    radio_str = f"{radio_km:.3f}".replace('.', '').zfill(4)

    # Directorio de salida
    BASE_DIR = Path(__file__).resolve().parent
    edges_dir = BASE_DIR / "graphs" / "edges" / "kdd" / f"{radio_str}_km"
    edges_dir.mkdir(parents=True, exist_ok=True)

    # Formato de ficheros de salida
    barrios = df["NEIGHBOURID"].unique()
    if len(barrios) == 1:
        edges_out = edges_dir / f"Valencia_kdd_edges_{radio_str}_{barrios[0]}.csv"
    else:
        edges_out = edges_dir / f"Valencia_kdd_edges_{radio_str}_all.csv"

    df_edges.to_csv(edges_out, index=False, encoding='utf-8')
    print(f"Aristas por vecindad exportadas a {edges_out}")

def exportar_aristas_similitud_caracteristicas(nodos, threshold):
    """
    La funcion crea un archivo CSV con las aristas de un grafo a partir
    de un archivo CSV de viviendas. Las aristas se crean entre nodos
    que tienen una similitud de caracteristicas mayor o igual al umbral especificado.

    Ademas, si un nodo no tiene vecinos con similitud suficiente, se conecta al nodo mas similar,
    asegurando asi que no existen nodos aislados.

    Parametros:
    nodos (str): Ruta al archivo CSV de entrada.
    threshold (float): Umbral de similitud para considerar vecinos.

    Returns:
    Ninguno
    """
    # Lectura del CSV
    df = pd.read_csv(nodos)

    # Atributos para calcular la similitud
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

    # Similitud euclidea
    nn = NearestNeighbors(radius=dist_threshold, metric='euclidean', n_jobs=-1)
    nn.fit(X)
    ind_array = nn.radius_neighbors(X, return_distance=False)

    # Creacion aristas en base a la similitud
    edge_rows = set()
    n = len(X)
    for i, neighbors in enumerate(ind_array):
        for j in neighbors:
            if i < j:
                dist = np.linalg.norm(X[i] - X[j]) / np.sqrt(n_features)
                similarity = 1 - dist
                edge_rows.add((i, j, similarity))

    # Detectar nodos aislados (sin vecinos)
    vecinos = {i for edge in edge_rows for i in edge[:2]}
    aislados = set(range(n)) - vecinos

    # Buscar el nodo mas similar para nodos aislados
    if aislados:
        dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2) / np.sqrt(n_features)
        for i in aislados:
            dists_i = dists[i].copy()
            dists_i[i] = np.inf
            j = np.argmin(dists_i)
            similarity = 1 - dists_i[j]
            edge_rows.add((min(i, j), max(i, j), similarity))

    df_edges = pd.DataFrame(list(edge_rows), columns=["source", "target", "SIMILARITY"])

    # Formateo del radio para nombres de archivo
    threshold_str = f"{threshold:.3f}".replace('.', '').zfill(4)

    # Salida de archivos
    BASE_DIR = Path(__file__).resolve().parent
    sim_dir = BASE_DIR / "graphs" / "edges" / "similarity" / f"{threshold_str}_sim"
    sim_dir.mkdir(parents=True, exist_ok=True)

    # Formato de ficheros de salida
    barrios = df["NEIGHBOURID"].unique()
    if len(barrios) == 1:
        edges_out = sim_dir / f"Valencia_similarity_edges_{threshold_str}_{barrios[0]}.csv"
    else:
        edges_out = sim_dir / f"Valencia_similarity_edges_{threshold_str}_all.csv"

    df_edges.to_csv(edges_out, index=False, encoding='utf-8')
    print(f"Aristas por similitud exportadas a {edges_out}")

def cargar_grafo(nodos, aristas):
    """
    La funcion carga un grafo a partir de dos archivos CSV:
    - Un archivo CSV con los nodos, que contiene las caracteristicas de cada vivienda.
    - Un archivo CSV con las aristas, que contiene las conexiones entre viviendas.

    Parametros:
    nodos (str): Ruta al archivo CSV de nodos.
    aristas (str): Ruta al archivo CSV de aristas.

    Returns:
    data (torch_geometric.data.Data): Objeto Data de PyTorch Geometric que representa el grafo.
    """
    # Lectura de los CSV
    nodes_df = pd.read_csv(nodos)
    edges_df = pd.read_csv(aristas)

    # Indicar nodos y caracteristicas
    node_features = nodes_df.drop(columns=["NODEID"]).values # Es identificador, no caracteristica
    x = torch.tensor(node_features, dtype=torch.float)

    # Indicar aristas del grafo
    edge_index = torch.tensor(edges_df[["source", "target"]].values.T, dtype=torch.long)

    # Peso de las aristas (si existe)
    edge_attr = None
    if edges_df.shape[1] > 2:
        weight_col = edges_df.columns[2]
        edge_attr = torch.tensor(edges_df[weight_col].values, dtype=torch.float).unsqueeze(1)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    else:
        data = Data(x=x, edge_index=edge_index)

    return data

def mostrar_grafo(data):
    """
    La funcion muestra un grafo utilizando NetworkX y Matplotlib a partir de un objeto Data de PyTorch Geometric.
    El grafo se dibuja con nodos y aristas, y si las aristas tienen peso, se muestran los pesos en rojo.

    Parametros:
    data (torch_geometric.data.Data): Objeto Data de PyTorch Geometric que representa el grafo.

    Returns:
    Ninguno
    """
    # Comprobar que el grafo tiene aristas
    if hasattr(data, 'edge_index'):
        edge_index = data.edge_index.cpu().numpy()
    else:
        raise ValueError("El objeto data no tiene atributo edge_index.")

    # Creacion del grafo de NetworkX
    G = nx.Graph()
    G.add_nodes_from(range(data.x.shape[0])) # Nodos
    edges = list(zip(edge_index[0], edge_index[1])) # Aristas
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        weights = data.edge_attr.cpu().numpy().flatten()
        for (u, v), w in zip(edges, weights):
            G.add_edge(u, v, weight=w)
    else:
        G.add_edges_from(edges)

    # Representacion del grafo
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=300, font_size=8)

    # Pesos de las aristas
    if nx.get_edge_attributes(G, 'weight'):
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()} # Redondeo a 2 decimales
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

    plt.show()
     
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    csv_in = BASE_DIR / "processed_data" / "Valencia_Sale_graph.csv"
    exportar_nodos(csv_in)

    # Ejecutar exportar_aristas_barrio para todos los CSV en graphs/nodes
    nodes_dir = BASE_DIR / "graphs" / "nodes"
    for csv_file in nodes_dir.glob("*.csv"):
        print(f"Procesando aristas para {csv_file.name}")
        exportar_aristas_barrio(csv_file)
        exportar_aristas_vecindad(csv_file, 0.05)
        exportar_aristas_vecindad(csv_file, 0.1)
        exportar_aristas_vecindad(csv_file, 0.15)
        exportar_aristas_vecindad(csv_file, 0.20)
        exportar_aristas_vecindad(csv_file, 0.25)
        exportar_aristas_vecindad(csv_file, 0.30)
        exportar_aristas_similitud_caracteristicas(csv_file, 0.99)
        exportar_aristas_similitud_caracteristicas(csv_file, 0.95)
        exportar_aristas_similitud_caracteristicas(csv_file, 0.90)
        exportar_aristas_similitud_caracteristicas(csv_file, 0.85)
        exportar_aristas_similitud_caracteristicas(csv_file, 0.80)
