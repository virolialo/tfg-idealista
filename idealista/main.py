import re
from pathlib import Path
from data_pipeline.graph_builder import cargar_grafo
from gnn_models.heterognn import train_node_regression_hetero
from gnn_models.gcn import train_node_regression
import math
import optuna

def evaluar_gcn_por_pares(nodos_dir, aristas_dir, price_index=0):
    """
    La funcion empareja archivos de nodos y aristas por el identificador de los grafos,
    para realizar lo siguiente por cada par:
    - Generar el grafo como un objeto de la clase Data de PyTorch Geometric
    - Buscar los hiperparametros optimos con Optuna
    - Entrenar el modelo GCN con los hiperparametros optimos
    - Guardar los resultados devueltos con las metricas MSE, MAE y R2

    Parametros:
    nodos_dir (str): Directorio donde se encuentran los archivos de nodos.
    aristas_dir (str): Directorio donde se encuentran los archivos de aristas.
    price_index (int): Indice de la columna de precios en los nodos, por defecto 0.

    Returns:
    Ninguno
    """
    nodos_dir = Path(nodos_dir) 
    aristas_dir = Path(aristas_dir)
    resultados = []

    # Directorio y fichero de salida
    edges_name = aristas_dir.name
    eval_dir = Path(__file__).resolve().parent / "evaluation" / "gcn"
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_file = eval_dir / f"model_gcn_evaluation_{edges_name}.txt"

    # Localizacion de pares por ID
    id_pattern = re.compile(r"_(\d+)\.csv$")

    def extraer_id(nombre):
        """
        La funcion extrae el ID de un nombre de archivo usando 
        una expresion regular.

        Parametros:
        nombre (str): Nombre del archivo del cual se extraera el ID.
        
        Returns:
        str: ID extraido del nombre del archivo
        None: Si no se encuentra un ID valido en el nombre del archivo.
        """
        m = id_pattern.search(nombre)
        return m.group(1) if m else None

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
        for id_ in ids_comunes:
            nodo_path = nodos_files[id_]
            arista_path = aristas_files[id_]
            try:
                data = cargar_grafo(str(nodo_path), str(arista_path))

                # Busqueda de hiperparametros optimos con Optuna
                def objective(trial):
                    hidden_channels = trial.suggest_int("hidden_channels", 8, 128)
                    dropout = trial.suggest_float("dropout", 0.0, 0.7)
                    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
                    epochs = trial.suggest_int("epochs", 50, 200)
                    mse, _, _ = train_node_regression(
                        data,
                        price_index=price_index,
                        epochs=epochs,
                        lr=lr,
                        hidden_channels=hidden_channels,
                        dropout=dropout
                    )
                    return mse if not math.isnan(mse) else float('inf')

                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=20, show_progress_bar=False)
                best_params = study.best_params

                # Entrenamiento final con los mejores hiperpararametros
                result = train_node_regression(
                    data,
                    price_index=price_index,
                    epochs=best_params["epochs"],
                    lr=best_params["lr"],
                    hidden_channels=best_params["hidden_channels"],
                    dropout=best_params["dropout"]
                )
                # Generacion de resultados
                if (
                    result is not None and
                    not any(math.isnan(x) for x in result)
                ):
                    mse, mae, r2 = result
                    resultados.append((float(mse), float(mae), float(r2)))
                    f.write(f"Grafo {id_}:\n")
                    f.write(f"Parametros usados: {best_params}\n")
                    f.write(f"Metricas: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}\n\n")
                    print(f"Evaluado ID {id_}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f} | Hiperparametros: {best_params}")
                else:
                    print(f"Saltando ID {id_} por metricas NaN o grafo pequeño.")
            except Exception as e:
                print(f"Error con ID {id_}: {e}")

        # Calculo de media de metricas de todos los pares validos
        if resultados:
            mean_mse = sum(r[0] for r in resultados) / len(resultados)
            mean_mae = sum(r[1] for r in resultados) / len(resultados)
            mean_r2 = sum(r[2] for r in resultados) / len(resultados)
            f.write(f"MEDIA DE METRICAS:\n")
            f.write(f"MSE={mean_mse:.4f}, MAE={mean_mae:.4f}, R2={mean_r2:.4f}\n")
            print(f"Media - MSE: {mean_mse:.4f}, MAE: {mean_mae:.4f}, R2: {mean_r2:.4f}")
        else:
            print("No se encontraron pares validos de nodos y aristas.")

def evaluar_hetero_por_pares(nodos_dir, aristas_dir1, aristas_dir2, price_index=0):
    """
    La funcion empareja archivos de nodos y aristas por el identificador de los grafos,
    para realizar lo siguiente por cada par:
    - Generar el grafo como un objeto de la clase Data de PyTorch Geometric
    - Buscar los hiperparametros optimos con Optuna
    - Entrenar el modelo HeteroGNN con los hiperparametros optimos
    - Guardar los resultados devueltos con las metricas MSE, MAE y R2

    Parametros:
    nodos_dir (str): Directorio donde se encuentran los archivos de nodos.
    aristas_dir (str): Directorio donde se encuentran los archivos de aristas.
    price_index (int): Indice de la columna de precios en los nodos, por defecto 0.

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
        for id_ in ids_comunes:
            nodo_path = nodos_files[id_]
            arista_path1 = aristas_files1[id_]
            arista_path2 = aristas_files2[id_]
            try:
                data = cargar_grafo(str(nodo_path), str(arista_path1), str(arista_path2))

                # Optuna para hiperparametros por par
                def objective(trial):
                    hidden_channels = trial.suggest_int("hidden_channels", 8, 128)
                    dropout = trial.suggest_float("dropout", 0.0, 0.7)
                    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
                    epochs = trial.suggest_int("epochs", 50, 200)
                    result = train_node_regression_hetero(
                        data,
                        price_index=price_index,
                        epochs=epochs,
                        lr=lr,
                        hidden_channels=hidden_channels,
                        dropout=dropout
                    )
                    mse = result[0] if result is not None else float('inf')
                    return mse if not math.isnan(mse) else float('inf')

                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=20, show_progress_bar=False)
                best_params = study.best_params

                # Entrenamiento final con los mejores hiperparametros
                result = train_node_regression_hetero(
                    data,
                    price_index=price_index,
                    epochs=best_params["epochs"],
                    lr=best_params["lr"],
                    hidden_channels=best_params["hidden_channels"],
                    dropout=best_params["dropout"]
                )
                # Generacion de resultados
                if (
                    result is not None and
                    not any(math.isnan(x) for x in result)
                ):
                    mse, mae, r2 = result
                    resultados.append((float(mse), float(mae), float(r2)))
                    f.write(f"Grafo {id_}:\n")
                    f.write(f"Parametros usados: {best_params}\n")
                    f.write(f"Metricas: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}\n\n")
                    print(f"Evaluado ID {id_}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f} | Hiperparametros: {best_params}")
                else:
                    print(f"Saltando ID {id_} por metricas NaN o grafo pequeño.")
            except Exception as e:
                print(f"Error con ID {id_}: {e}")

        # Calculo de media de metricas de todos los pares validos
        if resultados:
            mean_mse = sum(r[0] for r in resultados) / len(resultados)
            mean_mae = sum(r[1] for r in resultados) / len(resultados)
            mean_r2 = sum(r[2] for r in resultados) / len(resultados)
            f.write(f"MEDIA DE METRICAS:\n")
            f.write(f"MSE={mean_mse:.4f}, MAE={mean_mae:.4f}, R2={mean_r2:.4f}\n")
            print(f"Media - MSE: {mean_mse:.4f}, MAE: {mean_mae:.4f}, R2: {mean_r2:.4f}")
        else:
            print("No se encontraron pares validos de nodos y aristas.")

def crear_resumen_evaluation():
    """
    La funcion crea un resumen de los resultados de la evaluacion
    de los modelos GCN y HeteroGNN, extrayendo la ultima linea de cada fichero,
    que contiene la media de las metricas MSE, MAE y R2.

    Esta informacion se guarda en un fichero de texto llamado "resume.txt"
    en el directorio "evaluation".

    Parametros:
    Ninguno

    Returns:
    Ninguno
    """
    # Directorio y fichero de salida
    eval_dir = Path(__file__).resolve().parent / "evaluation"
    resumen_path = eval_dir / "resume.txt"
    lineas_resumen = []

    # Busqueda de fichero de evaluacion de modelos
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

    # Guardar el resumen en el fichero
    with open(resumen_path, "w", encoding="utf-8") as f:
        for linea in lineas_resumen:
            f.write(linea + "\n")
    print(f"Resumen guardado en {resumen_path}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent