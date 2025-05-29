import re
from pathlib import Path
from data_pipeline.graph_builder import cargar_grafo
from gnn_models.heterognn import train_node_regression_hetero
from gnn_models.gcn import train_node_regression
import math

def evaluar_gcn_por_pares(nodos_dir, aristas_dir, price_index=0, epochs=100, lr=0.01, hidden_channels=32, dropout=0.5):
    """
    La funcion empareja archivos de nodos y aristas por ID, ejecuta cargar_grafo (GCN) y train_node_regression,
    y guarda los resultados (ID, MSE, MAE, R2) en un fichero de texto en /evaluation, añadiendo la media al final.

    Parametros:
    nodos_dir: Ruta al directorio que contiene los archivos de nodos.
    aristas_dir: Ruta al directorio que contiene los archivos de aristas.
    price_index: Indice del precio en los nodos (por defecto 0).
    epochs: Numero de epocas para el entrenamiento (por defecto 100).
    lr: Tasa de aprendizaje para el optimizador (por defecto 0.01).
    hidden_channels: Numero de canales ocultos para las capas GCN (por defecto 32).
    dropout: Tasa de dropout para la regularizacion (por defecto 0.5).

    Returns:
    Ninguno    
    """
    nodos_dir = Path(nodos_dir) 
    aristas_dir = Path(aristas_dir)
    resultados = []

    # Ruta y nombre del fichero de salida
    edges_name = aristas_dir.name
    eval_dir = Path(__file__).resolve().parent / "evaluation" / "gcn"
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_file = eval_dir / f"model_gcn_evaluation_{edges_name}.txt"

    # Expresion para localizar pares por ID
    id_pattern = re.compile(r"_(\d+)\.csv$")

    def extraer_id(nombre):
        """
        La funcion extrae el ID de un nombre de archivo usando una expresion regular.

        Parametros:
        nombre: Nombre del archivo del cual se extrae el ID.

        Returns:
        id: El ID extraido del nombre del archivo, o None si no se encuentra.
        """
        m = id_pattern.search(nombre)
        return m.group(1) if m else None

    # Obtencion de IDs de los archivos de nodos y aristas
    nodos_files = {}
    for f in nodos_dir.glob("*.csv"):
        id_ = extraer_id(f.name)
        if id_ is not None:
            nodos_files[id_] = f

    aristas_files = {}
    for f in aristas_dir.glob("*.csv"):
        id_ = extraer_id(f.name)
        if id_ is not None:
            aristas_files[id_] = f

    # Emparejamiento por pares de nodos y aristas
    ids_comunes = sorted(set(nodos_files.keys()) & set(aristas_files.keys()), key=lambda x: int(x))
    print(f"IDs comunes encontrados: {len(ids_comunes)}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("ID,MSE,MAE,R2\n")
        for id_ in ids_comunes:
            nodo_path = nodos_files[id_]
            arista_path = aristas_files[id_]
            try:
                data = cargar_grafo(str(nodo_path), str(arista_path))
                result = train_node_regression(data, price_index, epochs, lr, hidden_channels, dropout)
                if (
                    result is not None and
                    not any(math.isnan(x) for x in result)
                ):
                    mse, mae, r2 = result
                    resultados.append((float(mse), float(mae), float(r2))) # Metricas por cada par
                    f.write(f"{id_},{mse:.4f},{mae:.4f},{r2:.4f}\n")
                    print(f"Evaluado ID {id_}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
                else:
                    print(f"Saltando ID {id_} por metricas NaN o grafo pequeño.")
            except Exception as e:
                print(f"Error con ID {id_}: {e}")

        # Calculo de media de metricas de todos los pares validos
        if resultados:
            mean_mse = sum(r[0] for r in resultados) / len(resultados)
            mean_mae = sum(r[1] for r in resultados) / len(resultados)
            mean_r2 = sum(r[2] for r in resultados) / len(resultados)
            f.write(f"MEDIA,{mean_mse:.4f},{mean_mae:.4f},{mean_r2:.4f}\n")
            print(f"Media - MSE: {mean_mse:.4f}, MAE: {mean_mae:.4f}, R2: {mean_r2:.4f}")
        else:
            print("No se encontraron pares validos de nodos y aristas.")

def evaluar_hetero_por_pares(nodos_dir, aristas_dir1, aristas_dir2, price_index=0, epochs=100, lr=0.01, hidden_channels=32, dropout=0.5):
    """
    La funcion empareja archivos de nodos y aristas por ID, ejecuta cargar_grafo (HeteroGNN) y train_node_regression_hetero,
    y guarda los resultados (ID, MSE, MAE, R2) en un fichero de texto en /evaluation, añadiendo la media al final.

    Parametros:
    nodos_dir: Ruta al directorio que contiene los archivos de nodos.
    aristas_dir1: Ruta al directorio que contiene los archivos de aristas del primer grafo.
    aristas_dir2: Ruta al directorio que contiene los archivos de aristas del segundo grafo.
    price_index: Indice del precio en los nodos (por defecto 0).
    epochs: Numero de epocas para el entrenamiento (por defecto 100).
    lr: Tasa de aprendizaje para el optimizador (por defecto 0.01).
    hidden_channels: Numero de canales ocultos para las capas GCN (por defecto 32).
    dropout: Tasa de dropout para la regularizacion (por defecto 0.5).

    Returns:
    Ninguno
    """
    nodos_dir = Path(nodos_dir)
    aristas_dir1 = Path(aristas_dir1)
    aristas_dir2 = Path(aristas_dir2)
    resultados = []

    # Ruta y nombre del fichero de salida
    edges_name1 = aristas_dir1.name
    edges_name2 = aristas_dir2.name
    eval_dir = Path(__file__).resolve().parent / "evaluation" / "heterognn"
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_file = eval_dir / f"model_hetero_evaluation_{edges_name1}_{edges_name2}.txt"

    # Expresion para localizar pares por ID
    id_pattern = re.compile(r"_(\d+)\.csv$")

    def extraer_id(nombre):
        m = id_pattern.search(nombre)
        return m.group(1) if m else None

    # Obtencion de IDs de los archivos de nodos y aristas
    nodos_files = {}
    for f in nodos_dir.glob("*.csv"):
        id_ = extraer_id(f.name)
        if id_ is not None:
            nodos_files[id_] = f

    aristas_files1 = {}
    for f in aristas_dir1.glob("*.csv"):
        id_ = extraer_id(f.name)
        if id_ is not None:
            aristas_files1[id_] = f

    aristas_files2 = {}
    for f in aristas_dir2.glob("*.csv"):
        id_ = extraer_id(f.name)
        if id_ is not None:
            aristas_files2[id_] = f

    # Emparejamiento por pares de nodos y aristas
    ids_comunes = sorted(set(nodos_files.keys()) & set(aristas_files1.keys()) & set(aristas_files2.keys()), key=lambda x: int(x))
    print(f"IDs comunes encontrados: {len(ids_comunes)}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("ID,MSE,MAE,R2\n")
        for id_ in ids_comunes:
            nodo_path = nodos_files[id_]
            arista_path1 = aristas_files1[id_]
            arista_path2 = aristas_files2[id_]
            try:
                data = cargar_grafo(str(nodo_path), str(arista_path1), str(arista_path2))
                result = train_node_regression_hetero(data, price_index, epochs, lr, hidden_channels, dropout)
                if (
                    result is not None and
                    not any(math.isnan(x) for x in result)
                ):
                    mse, mae, r2 = result
                    resultados.append((float(mse), float(mae), float(r2))) # Metricas por cada par
                    f.write(f"{id_},{mse:.4f},{mae:.4f},{r2:.4f}\n")
                    print(f"Evaluado ID {id_}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
                else:
                    print(f"Saltando ID {id_} por metricas NaN o grafo pequeño.")
            except Exception as e:
                print(f"Error con ID {id_}: {e}")

        # Calculo de media de metricas de todos los pares validos
        if resultados:
            mean_mse = sum(r[0] for r in resultados) / len(resultados)
            mean_mae = sum(r[1] for r in resultados) / len(resultados)
            mean_r2 = sum(r[2] for r in resultados) / len(resultados)
            f.write(f"MEDIA,{mean_mse:.4f},{mean_mae:.4f},{mean_r2:.4f}\n")
            print(f"Media - MSE: {mean_mse:.4f}, MAE: {mean_mae:.4f}, R2: {mean_r2:.4f}")
        else:
            print("No se encontraron pares validos de nodos y aristas.")

def crear_resumen_evaluation():
    """
    La funcion crea un resumen de los resultados de evaluacion de los modelos GCN y HeteroGNN,
    buscando los archivos de texto en el directorio /evaluation y guardando la ultima linea no vacia de cada uno
    en un fichero resumen.txt.

    Parametros:
    Ninguno

    Returns:
    Ninguno
    """
    # Ruta y fichero de salida del resumen
    eval_dir = Path(__file__).resolve().parent / "evaluation"
    resumen_path = eval_dir / "resume.txt"
    lineas_resumen = []

    for subdir in eval_dir.iterdir():
        if not subdir.is_dir():
            continue
        for file in subdir.glob("*.txt"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    last_line = ""
                    for line in reversed(lines):
                        if line.strip():
                            last_line = line.strip()
                            break
                    if last_line:
                        lineas_resumen.append(f"{file.name}: {last_line}")
            except Exception as e:
                print(f"Error leyendo {file}: {e}")

    # Escritura del resumen en el fichero
    with open(resumen_path, "w", encoding="utf-8") as f:
        for linea in lineas_resumen:
            f.write(linea + "\n")
    print(f"Resumen guardado en {resumen_path}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
            