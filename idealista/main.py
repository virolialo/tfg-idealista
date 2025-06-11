import re
from pathlib import Path
from data_pipeline.graph_builder import cargar_grafo
from gnn_models.heterognn import train_node_regression_hetero
from gnn_models.gcn import train_node_regression
from gnn_models.gcn import GCN
from gnn_models.heterognn import HeteroGNN
import math
import optuna
import torch

def evaluar_gcn_por_pares(nodos_dir, aristas_dir, price_index=0):
    """
    La funcion empareja archivos de nodos y aristas por el identificador de los grafos,
    para realizar lo siguiente por cada par:
    - Generar el grafo como un objeto de la clase Data de PyTorch Geometric.
    - Buscar los hiperparametros optimos de entrenamiento con Optuna.
    - Entrenar el modelo GCN con los hiperparametros devueltos.
    - Evaluar el modelo y guardar los resultados devueltos con las metricas MSE, MAE y R2.
    - Guardar los pesos del modelo entrenado en un archivo .pt por cada grafo.

    Parametros:
    nodos_dir (str): Directorio donde se encuentran los archivos de nodos.
    aristas_dir (str): Directorio donde se encuentran los archivos de aristas.
    price_index (int): Indice de la columna de precios en los nodos, por defecto 0.

    Returns:
    Ninguno.

    Raises:
    FileNotFoundError: Si alguno de los archivos de nodos o aristas no existe.
    KeyError: Si faltan columnas requeridas en los archivos de nodos o aristas.
    ValueError: Si los datos de entrada no cumplen los requisitos esperados (por ejemplo, dimensiones incompatibles).
    RuntimeError: Si ocurre un error durante el entrenamiento o evaluación del modelo (por ejemplo, problemas de CUDA o PyTorch).
    Exception: Si ocurre cualquier otro error durante la carga, entrenamiento, evaluación o guardado de modelos y resultados.
    """
    nodos_dir = Path(nodos_dir) 
    aristas_dir = Path(aristas_dir)
    resultados = []

    # Directorio y fichero de salida
    edges_name = aristas_dir.name
    eval_dir = Path(__file__).resolve().parent / "evaluation" / "gcn"
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_file = eval_dir / f"model_gcn_evaluation_{edges_name}.txt"
    pesos_dir = eval_dir / "pesos"
    pesos_dir.mkdir(parents=True, exist_ok=True)

    id_pattern = re.compile(r"_(\d+)\.csv$")

    # Obtencion de pares de nodos y aristas
    def extraer_id(nombre):
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
                    num_layers = trial.suggest_int("num_layers", 2, 5)
                    dropout = trial.suggest_float("dropout", 0.0, 0.7)
                    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
                    epochs = trial.suggest_int("epochs", 50, 200)
                    mse, _, _ = train_node_regression(
                        data,
                        price_index=price_index,
                        epochs=epochs,
                        lr=lr,
                        hidden_channels=hidden_channels,
                        num_layers=num_layers,
                        dropout=dropout
                    )
                    return mse if not math.isnan(mse) else float('inf')

                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=10, show_progress_bar=False)
                best_params = study.best_params

                # Entrenamiento por medio de los mejores hiperparametros
                result = train_node_regression(
                    data,
                    price_index=price_index,
                    epochs=best_params["epochs"],
                    lr=best_params["lr"],
                    hidden_channels=best_params["hidden_channels"],
                    num_layers=best_params["num_layers"],
                    dropout=best_params["dropout"]
                )
                if result is not None and not any(math.isnan(x) for x in result):
                    mse, mae, r2 = result
                    resultados.append((float(mse), float(mae), float(r2)))
                    # Entrena el modelo final para guardar los pesos
                    model = None
                    try:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model = GCN(
                            data.num_features,
                            best_params["hidden_channels"],
                            best_params["num_layers"],
                            best_params["dropout"]
                        ).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
                        data_device = data.to(device)
                        y = data_device.x[:, price_index]
                        idx = torch.randperm(data_device.num_nodes)
                        split = int(0.8 * data_device.num_nodes)
                        train_idx = idx[:split]
                        for epoch in range(best_params["epochs"]):
                            model.train()
                            optimizer.zero_grad()
                            out = model(data_device)
                            loss = torch.nn.functional.mse_loss(out[train_idx], y[train_idx])
                            loss.backward()
                            optimizer.step()
                        # Pesos del entrenamiento
                        pesos_path = pesos_dir / f"pesos_gcn_{id_}.pt"
                        torch.save(model.state_dict(), pesos_path)
                    except Exception as e:
                        print(f"Error guardando pesos para ID {id_}: {e}")
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
    - Generar el grafo como un objeto de la clase Data de PyTorch Geometric.
    - Buscar los hiperparametros optimos con Optuna.
    - Entrenar el modelo HeteroGNN con los hiperparametros optimos.
    - Guardar los resultados devueltos con las metricas MSE, MAE y R2.
    - Guardar los pesos del modelo entrenado en un archivo .pt por cada grafo.

    Parametros:
    nodos_dir (str): Directorio donde se encuentran los archivos de nodos.
    aristas_dir (str): Directorio donde se encuentran los archivos de aristas.
    price_index (int): Indice de la columna de precios en los nodos, por defecto 0.

    Returns:
    Ninguno.

    Raises:
    FileNotFoundError: Si alguno de los archivos de nodos o aristas no existe.
    KeyError: Si faltan columnas requeridas en los archivos de nodos o aristas.
    ValueError: Si los datos de entrada no cumplen los requisitos esperados (por ejemplo, dimensiones incompatibles).
    RuntimeError: Si ocurre un error durante el entrenamiento o evaluación del modelo (por ejemplo, problemas de CUDA o PyTorch).
    Exception: Si ocurre cualquier otro error durante la carga, entrenamiento, evaluación o guardado de modelos y resultados.
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
    pesos_dir = eval_dir / "pesos"
    pesos_dir.mkdir(parents=True, exist_ok=True)

    id_pattern = re.compile(r"_(\d+)\.csv$")

    # Obtencion de pares de nodos y aristas
    def extraer_id(nombre):
        m = id_pattern.search(nombre)
        return m.group(1) if m else None

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

                # Busqueda de hiperparametros optimos con Optuna
                def objective(trial):
                    hidden_channels = trial.suggest_int("hidden_channels", 8, 128)
                    num_layers = trial.suggest_int("num_layers", 2, 5)
                    dropout = trial.suggest_float("dropout", 0.0, 0.7)
                    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
                    epochs = trial.suggest_int("epochs", 50, 200)
                    result = train_node_regression_hetero(
                        data,
                        price_index=price_index,
                        epochs=epochs,
                        lr=lr,
                        hidden_channels=hidden_channels,
                        num_layers=num_layers,
                        dropout=dropout
                    )
                    mse = result[0] if result is not None else float('inf')
                    return mse if not math.isnan(mse) else float('inf')

                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=10, show_progress_bar=False)
                best_params = study.best_params

                # Entrenamiento por medio de los mejores hiperparametros
                result = train_node_regression_hetero(
                    data,
                    price_index=price_index,
                    epochs=best_params["epochs"],
                    lr=best_params["lr"],
                    hidden_channels=best_params["hidden_channels"],
                    num_layers=best_params["num_layers"],
                    dropout=best_params["dropout"]
                )
                if (
                    result is not None and
                    not any(math.isnan(x) for x in result)
                ):
                    mse, mae, r2 = result
                    resultados.append((float(mse), float(mae), float(r2)))
                    # Entrena el modelo final para guardar los pesos
                    model = None
                    try:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model = HeteroGNN(
                            data.metadata(),
                            best_params["hidden_channels"],
                            best_params["num_layers"],
                            best_params["dropout"]
                        ).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
                        data_device = data.to(device)
                        y = data_device['house'].x[:, price_index]
                        idx = torch.randperm(data_device['house'].num_nodes)
                        split = int(0.8 * data_device['house'].num_nodes)
                        train_idx = idx[:split]
                        for epoch in range(best_params["epochs"]):
                            model.train()
                            optimizer.zero_grad()
                            out = model(data_device)
                            loss = torch.nn.functional.mse_loss(out[train_idx], y[train_idx])
                            loss.backward()
                            optimizer.step()
                        # Pesos del entrenamiento
                        pesos_path = pesos_dir / f"pesos_hetero_{id_}.pt"
                        torch.save(model.state_dict(), pesos_path)
                    except Exception as e:
                        print(f"Error guardando pesos para ID {id_}: {e}")
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
    Ninguno.

    Returns:
    Ninguno.

    Raises:
    FileNotFoundError: Si el directorio "evaluation" no existe.
    PermissionError: Si no se tiene permiso para leer o escribir archivos en el directorio.
    Exception: Si ocurre cualquier otro error al leer los archivos de evaluación o al escribir el resumen.
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

    # Rutas concretas para probar las funciones
    nodes_dir = BASE_DIR / "data_pipeline" / "graphs" / "nodes"
    edges_dir1 = BASE_DIR / "data_pipeline" / "graphs" / "edges" / "kdd" / "0050_km"
    edges_dir2 = BASE_DIR / "data_pipeline" / "graphs" / "edges" / "similarity" / "0990_sim"

    # Ejemplo de llamada a la funcion para evaluar GCN por pares
    evaluar_gcn_por_pares(nodes_dir, edges_dir1, price_index=0)

    # Ejemplo de llamada a la funcion para evaluar HeteroGNN por pares
    evaluar_hetero_por_pares(nodes_dir, edges_dir1, edges_dir2, price_index=0)