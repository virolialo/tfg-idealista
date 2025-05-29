import re
from pathlib import Path
from data_pipeline.graph_builder import cargar_grafo
from gnn_models.gcn import train_node_regression

def evaluar_gcn_por_pares(nodos_dir, aristas_dir, price_index=0, epochs=100, lr=0.01, hidden_channels=32, dropout=0.5):
    """
    Empareja archivos de nodos y aristas por ID, ejecuta cargar_grafo y train_node_regression,
    y guarda los resultados (ID, MSE, MAE, R2) en un fichero de texto en /evaluation.
    Añade la media al final. El nombre del fichero es model_gcn_evaluation_{edges}.txt
    """
    import math

    nodos_dir = Path(nodos_dir)
    aristas_dir = Path(aristas_dir)
    resultados = []

    # Obtener nombre de la subcarpeta de aristas para el nombre del fichero
    edges_name = aristas_dir.name
    eval_dir = Path(__file__).resolve().parent / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_file = eval_dir / f"model_gcn_evaluation_{edges_name}.txt"

    # Expresión regular para extraer el ID al final antes de .csv
    id_pattern = re.compile(r"_(\d+)\.csv$")

    def extraer_id(nombre):
        m = id_pattern.search(nombre)
        return m.group(1) if m else None

    # Crear diccionarios {id: path}
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

    # Emparejar por ID
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
                # Solo guardar si no es None y no hay nan en las métricas
                if (
                    result is not None and
                    not any(math.isnan(x) for x in result)
                ):
                    mse, mae, r2 = result
                    resultados.append((float(mse), float(mae), float(r2)))
                    f.write(f"{id_},{mse:.4f},{mae:.4f},{r2:.4f}\n")
                    print(f"Evaluado ID {id_}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
                else:
                    print(f"Saltando ID {id_} por métricas NaN o grafo pequeño.")
            except Exception as e:
                print(f"Error con ID {id_}: {e}")

        # Calcular medias solo con resultados válidos
        if resultados:
            mean_mse = sum(r[0] for r in resultados) / len(resultados)
            mean_mae = sum(r[1] for r in resultados) / len(resultados)
            mean_r2 = sum(r[2] for r in resultados) / len(resultados)
            f.write(f"MEDIA,{mean_mse:.4f},{mean_mae:.4f},{mean_r2:.4f}\n")
            print(f"Media - MSE: {mean_mse:.4f}, MAE: {mean_mae:.4f}, R2: {mean_r2:.4f}")
        else:
            print("No se encontraron pares válidos de nodos y aristas.")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    nodos_dir = BASE_DIR / "data_pipeline" / "graphs" / "nodes"
    aristas_dir = BASE_DIR / "data_pipeline" / "graphs" / "edges" / "similarity" / "0990_sim"

    evaluar_gcn_por_pares(
        nodos_dir=nodos_dir,
        aristas_dir=aristas_dir,
        price_index=0,
        epochs=100,
        lr=0.01,
        hidden_channels=32,
        dropout=0.5
    )
            