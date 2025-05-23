import pandas as pd
from neo4j import GraphDatabase
from django.conf import settings
import os
import sys
import django

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'idealista.settings')
django.setup()

def upload_graph_to_neo4j(uri, user, password, nodes_csv, edges_csv):
    # Leer los CSV
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)
    
    # Conectar a Neo4j
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        # Crear nodos
        for _, row in nodes_df.iterrows():
            props = row.to_dict()
            node_id = props.pop('NODEID')
            # Cypher para crear nodo con propiedades
            cypher = (
                "MERGE (n:Node {NODEID: $node_id}) "
                "SET n += $props"
            )
            print(f"Subiendo nodo: {node_id} con propiedades: {props}")
            session.run(cypher, node_id=node_id, props=props)
        
        # Crear aristas
        for _, row in edges_df.iterrows():
            source = row['source']
            target = row['target']
            weight = row['weight']
            cypher = (
                "MATCH (a:Node {NODEID: $source}), (b:Node {NODEID: $target}) "
                "MERGE (a)-[r:CONNECTED {weight: $weight}]->(b)"
            )
            print(f"Subiendo arista: {source} -> {target} (peso: {weight})")
            session.run(cypher, source=source, target=target, weight=weight)
    driver.close()
    print("¡Grafo subido a Neo4j!")

if __name__ == "__main__":
    # Configura aquí tus rutas y credenciales
    uri = settings.NEO4J_URI
    user = settings.NEO4J_USER
    password = settings.NEO4J_PASSWORD
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nodes_csv = os.path.join(BASE_DIR, "data_pipeline", "graphs", "Valencia_nodes.csv")
    edges_csv = os.path.join(BASE_DIR, "data_pipeline", "graphs", "Valencia_similarity_edges_0990.csv")

    upload_graph_to_neo4j(uri, user, password, nodes_csv, edges_csv)