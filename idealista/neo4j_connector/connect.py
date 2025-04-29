import os
import sys
import django
from py2neo import Graph
from django.conf import settings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'idealista.settings')
django.setup()

class Neo4jConnection:
    """Clase para manejar la conexión a la base de datos Neo4j."""
    
    def __init__(self):
        """Inicializa la conexión con la base de datos Neo4j."""
        self.graph = Graph(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))

    def test_connection(self):
        """Prueba la conexión ejecutando una consulta simple."""
        try:
            result = self.graph.run("RETURN 'Conexión exitosa' AS mensaje").data()
            return result
        except Exception as e:
            print(f"Error al conectar con Neo4j: {e}")
            return None

if __name__ == "__main__":
    conn = Neo4jConnection()
    result = conn.test_connection()
    if result:
        for record in result:
            print(record["mensaje"])