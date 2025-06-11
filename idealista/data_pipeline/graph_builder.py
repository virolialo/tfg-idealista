import pandas as pd
from pathlib import Path
import itertools
import numpy as np
from sklearn.neighbors import NearestNeighbors, BallTree
import torch
from torch_geometric.data import Data, HeteroData
import matplotlib.pyplot as plt
import networkx as nx

def exportar_nodos(csv_entrada):
    """
    La funcion exporta los nodos de un grafo a partir de un archivo CSV de viviendas.
    Guarda los nodos en archivos CSV separados por barrio en el directorio graphs/nodes.

    Parametros:
    csv_entrada (str): Ruta al archivo CSV de entrada que contiene las viviendas.

    Returns:
    Ninguno.

    Raises:
    FileNotFoundError: Si el archivo CSV de entrada no existe.
    KeyError: Si falta alguna de las columnas requeridas en el archivo CSV.
    Exception: Si ocurre un error al escribir los archivos de nodos.
    """
    node_cols = [
        "UNITPRICE", "CONSTRUCTEDAREA", "ROOMNUMBER", "BATHNUMBER",
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

def exportar_aristas_barrio(nodos):
    """
    La funcion crea un archivo CSV con las aristas de un grafo a partir
    de un archivo CSV de nodos. Las aristas se crean entre nodos que pertenecen
    al mismo barrio, considerando que son vecinos.

    Parametros:
    nodos (str): Ruta al archivo CSV de nodos.

    Returns:
    Ninguno.

    Raises:
    FileNotFoundError: Si el archivo de nodos no existe.
    KeyError: Si la columna 'NEIGHBOURID' no esta presente en el archivo de nodos.
    Exception: Si ocurre un error al escribir los archivos de aristas.
    """
    # Lectura del CSV
    df = pd.read_csv(nodos)
    edge_rows = []
    barrios = df["NEIGHBOURID"].unique()

    # Agrupar por barrio y exportar solo por barrio
    for barrio in barrios:
        group = df[df["NEIGHBOURID"] == barrio]
        idx = group.index.to_numpy()
        if len(idx) > 1:
            comb = np.array(list(itertools.combinations(idx, 2)))
            edge_rows = [{"source": i, "target": j} for i, j in comb]
            df_edges = pd.DataFrame(edge_rows)

            # Directorio de salida
            BASE_DIR = Path(__file__).resolve().parent
            edges_dir = BASE_DIR / "graphs" / "edges" / "neighbour"
            edges_dir.mkdir(parents=True, exist_ok=True)

            # Ficheros de salida por barrio
            edges_out = edges_dir / f"Valencia_neighbour_edges_{barrio}.csv"
            df_edges.to_csv(edges_out, index=False, encoding='utf-8')
            print(f"Aristas exportadas a {edges_out}")
        else:
            print(f"Barrio {barrio} no tiene suficientes nodos para crear aristas.")

def exportar_aristas_vecindad(nodos, radio_km):
    """
    La funcion crea un archivo CSV con las aristas de un grafo a partir de un archivo CSV de nodos.
    Las aristas se crean entre nodos que estan dentro de un radio especificado en kilometros.
    Si un nodo no tiene vecinos dentro del radio, se conecta al nodo mas cercano.

    Parametros:
    nodos (str): Ruta al archivo CSV de nodos.
    radio_km (float): Radio en kilometros para considerar vecinos.

    Returns:
    Ninguno.

    Raises:
    FileNotFoundError: Si el archivo de nodos no existe.
    KeyError: Si faltan las columnas 'LATITUDE', 'LONGITUDE' o 'NEIGHBOURID' en el archivo de nodos.
    Exception: Si ocurre un error al escribir los archivos de aristas.
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

    # Deteccion de nodos aislados
    vecinos = {i for edge in edge_rows for i in edge[:2]}
    aislados = set(range(n)) - vecinos

    # Busqueda del vecino mas cercano en nodos aislados
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

    # Ficheros de salida por barrio
    barrios = df["NEIGHBOURID"].unique()
    if len(barrios) == 1:
        edges_out = edges_dir / f"Valencia_kdd_edges_{radio_str}_{barrios[0]}.csv"
        df_edges.to_csv(edges_out, index=False, encoding='utf-8')
        print(f"Aristas por vecindad exportadas a {edges_out}")
    else:
        for barrio in barrios:
            idxs = df[df["NEIGHBOURID"] == barrio].index.values
            mask = df_edges["source"].isin(idxs) & df_edges["target"].isin(idxs)
            df_edges_barrio = df_edges[mask]
            if not df_edges_barrio.empty:
                edges_out = edges_dir / f"Valencia_kdd_edges_{radio_str}_{barrio}.csv"
                df_edges_barrio.to_csv(edges_out, index=False, encoding='utf-8')
                print(f"Aristas por vecindad exportadas a {edges_out}")

def exportar_aristas_similitud_caracteristicas(nodos, threshold):
    """
    La funcion crea un archivo CSV con las aristas de un grafo a partir de un archivo CSV de nodos.
    Las aristas se crean entre nodos que son similares en base a un umbral de similitud.
    Si un nodo no tiene vecinos similares, se conecta al nodo mas similar.

    Parametros:
    nodos (str): Ruta al archivo CSV de nodos.
    threshold (float): Umbral de similitud entre 0 y 1.
    
    Returns:
    Ninguno

    Raises:
    FileNotFoundError: Si el archivo de nodos no existe.
    KeyError: Si faltan las columnas requeridas para calcular la similitud o la columna 'NEIGHBOURID'.
    Exception: Si ocurre un error al escribir los archivos de aristas.
    """
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

    # Formateo del threshold para nombres de archivo
    threshold_str = f"{threshold:.3f}".replace('.', '').zfill(4)

    # Salida de archivos
    BASE_DIR = Path(__file__).resolve().parent
    sim_dir = BASE_DIR / "graphs" / "edges" / "similarity" / f"{threshold_str}_sim"
    sim_dir.mkdir(parents=True, exist_ok=True)

    # Ficheros de salida por barrio
    barrios = df["NEIGHBOURID"].unique()
    if len(barrios) == 1:
        edges_out = sim_dir / f"Valencia_similarity_edges_{threshold_str}_{barrios[0]}.csv"
        df_edges.to_csv(edges_out, index=False, encoding='utf-8')
        print(f"Aristas por similitud exportadas a {edges_out}")
    else:
        for barrio in barrios:
            idxs = df[df["NEIGHBOURID"] == barrio].index.values
            mask = df_edges["source"].isin(idxs) & df_edges["target"].isin(idxs)
            df_edges_barrio = df_edges[mask]
            if not df_edges_barrio.empty:
                edges_out = sim_dir / f"Valencia_similarity_edges_{threshold_str}_{barrio}.csv"
                df_edges_barrio.to_csv(edges_out, index=False, encoding='utf-8')
                print(f"Aristas por similitud exportadas a {edges_out}")

def cargar_grafo(nodos, aristas, aristas_extra=None):
    """
    Carga un grafo a partir de un archivo de nodos y uno o dos archivos de aristas.
    Si aristas_extra es None, devuelve un Data homogeneo.
    Si aristas_extra no es None, devuelve un HeteroData con dos tipos de aristas.
    El nombre del tipo de arista sera 'vivienda_{caracteristica}', donde 'caracteristica' es la cabecera de la columna de peso si existe, o 'simple' si no hay peso.

    Parametros:
    nodos (str): Ruta al archivo CSV de nodos.
    aristas (str): Ruta al archivo CSV de aristas principal.
    aristas_extra (str, opcional): Ruta al archivo CSV de aristas secundario.

    Returns:
    data (torch_geometric.data.Data o HeteroData): Grafo homogeneo o heterogeneo.

    Raises:
    FileNotFoundError: Si alguno de los archivos CSV no existe.
    KeyError: Si faltan columnas requeridas en los archivos de nodos o aristas (por ejemplo, 'NODEID', 'source', 'target').
    Exception: Si ocurre un error al convertir los datos a tensores o al construir el grafo.
    """
    # Lectura de los CSV
    nodes_df = pd.read_csv(nodos)
    edges_df = pd.read_csv(aristas)

    # Indicar nodos y caracteristicas
    node_features = nodes_df.drop(columns=["NODEID"]).values
    x = torch.tensor(node_features, dtype=torch.float)

    # Primer tipo de aristas
    edge_index1 = torch.tensor(edges_df[["source", "target"]].values.T, dtype=torch.long)
    if edges_df.shape[1] > 2:
        weight_col1 = edges_df.columns[2]
        edge_attr1 = torch.tensor(edges_df[weight_col1].values, dtype=torch.float).unsqueeze(1)
        nombre_arista1 = f"{weight_col1.lower()}"
    else:
        edge_attr1 = None
        nombre_arista1 = "vivienda_simple"

    if aristas_extra is None:
        if edge_attr1 is not None:
            data = Data(x=x, edge_index=edge_index1, edge_attr=edge_attr1)
        else:
            data = Data(x=x, edge_index=edge_index1)
        return data

    # Segundo tipo de aristas
    edges_df2 = pd.read_csv(aristas_extra)
    edge_index2 = torch.tensor(edges_df2[["source", "target"]].values.T, dtype=torch.long)
    if edges_df2.shape[1] > 2:
        weight_col2 = edges_df2.columns[2]
        edge_attr2 = torch.tensor(edges_df2[weight_col2].values, dtype=torch.float).unsqueeze(1)
        nombre_arista2 = f"{weight_col2.lower()}"
    else:
        edge_attr2 = None
        nombre_arista2 = "vivienda_simple"

    data = HeteroData()
    data['house'].x = x
    data['house', nombre_arista1, 'house'].edge_index = edge_index1
    if edge_attr1 is not None:
        data['house', nombre_arista1, 'house'].edge_attr = edge_attr1
    data['house', nombre_arista2, 'house'].edge_index = edge_index2
    if edge_attr2 is not None:
        data['house', nombre_arista2, 'house'].edge_attr = edge_attr2

    return data

def mostrar_grafo(data):
    """
    Muestra un grafo (Data o HeteroData) usando NetworkX y Matplotlib.
    Si es HeteroData, muestra cada tipo de arista en un color diferente.

    Parametros:
    data (torch_geometric.data.Data o HeteroData): Grafo a mostrar.

    Returns:
    Ninguno

    Raises:
    ValueError: Si el objeto data no es de tipo Data ni HeteroData.
    Exception: Si ocurre un error al procesar o visualizar el grafo.
    """
    if isinstance(data, Data):
        # Grafo homogeneo
        edge_index = data.edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(data.x.shape[0]))
        edges = list(zip(edge_index[0], edge_index[1]))
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            weights = data.edge_attr.cpu().numpy().flatten()
            for (u, v), w in zip(edges, weights):
                G.add_edge(u, v, weight=w)
        else:
            G.add_edges_from(edges)
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=300, font_size=8)
        if nx.get_edge_attributes(G, 'weight'):
            edge_labels = nx.get_edge_attributes(G, 'weight')
            edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
        plt.show()
    elif isinstance(data, HeteroData):
        # Grafo heterogeneo 
        G = nx.Graph()
        num_nodes = data['house'].x.shape[0]
        G.add_nodes_from(range(num_nodes))
        colors = ['gray', 'orange', 'green', 'blue', 'red', 'purple']
        edge_types = list(data.edge_types)
        for idx, edge_type in enumerate(edge_types):
            edge_index = data[edge_type].edge_index.cpu().numpy()
            edges = list(zip(edge_index[0], edge_index[1]))
            color = colors[idx % len(colors)]
            if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                weights = data[edge_type].edge_attr.cpu().numpy().flatten()
                for (u, v), w in zip(edges, weights):
                    G.add_edge(u, v, weight=w, color=color)
            else:
                for (u, v) in edges:
                    G.add_edge(u, v, color=color)
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(G, seed=42)

        # Diferencia los tipos de aristas por color
        for idx, edge_type in enumerate(edge_types):
            color = colors[idx % len(colors)]
            edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get('color') == color]
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, label=str(edge_type))
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=300)
        nx.draw_networkx_labels(G, pos, font_size=8)

        # Dibuja pesos si existen
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True) if 'weight' in d}
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
        plt.legend()
        plt.show()
    else:
        raise ValueError("El objeto data debe ser de tipo Data o HeteroData.")
     
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    csv_in = BASE_DIR / "processed_data" / "Valencia_Sale_graph.csv"
    exportar_nodos(csv_in)

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
    print("Exportacion de aristas completada.")