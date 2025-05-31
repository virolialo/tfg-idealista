import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv
from torch_geometric.data import HeteroData
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=32, dropout=0.5):
        """
        Modelo HeteroGNN para regresion de nodos en grafos heterogeneos.

        Parametros:
        metadata: Metadata del grafo heterogeneo.
        hidden_channels: Numero de canales ocultos para las capas GCN.
        dropout: Tasa de dropout para la regularizacion.

        Returns:
        Ninguno
        """
        # Inicializa la clase HeteroGNN
        super().__init__()
        self.conv1 = HeteroConv({
            edge_type: GCNConv(-1, hidden_channels)
            for edge_type in metadata[1]
        }, aggr='sum')
        self.lin = torch.nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, data):
        """
        La funcion realiza el paso hacia adelante del modelo HeteroGNN.

        Parametros:
        data: HeteroData que contiene el grafo y las caracteristicas de los nodos.

        Returns:
        x: Salida del modelo, que es una prediccion de precios de nodos.
        """
        x_dict = self.conv1(data.x_dict, data.edge_index_dict)
        x = x_dict['house']
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x.squeeze()

def train_node_regression_hetero(data, price_index=0, epochs=100, lr=0.01, hidden_channels=32, dropout=0.5):
    """
    La funcion entrena un modelo HeteroGNN para la regresion de precios de nodos en un grafo heterogeneo.

    Parametros:
    data: HeteroData que contiene el grafo y las caracteristicas de los nodos.
    price_index: Indice de la caracteristica del precio en los nodos.
    epochs: Numero de epocas para el entrenamiento.
    lr: Tasa de aprendizaje para el optimizador.
    hidden_channels: Numero de canales ocultos para las capas GCN.
    dropout: Tasa de dropout para la regularizacion.

    Returns:
    mse: Error cuadratico medio en el conjunto de prueba.
    mae: Error absoluto medio en el conjunto de prueba.
    r2: Coeficiente de determinacion R^2 en el conjunto de prueba.
    """
    if data['house'].num_nodes < 10:
        print(f"Saltando grafo con solo {data['house'].num_nodes} nodos.")
        return None

    # Configuracion del modelo y optimizador
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeteroGNN(data.metadata(), hidden_channels, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data = data.to(device)

    num_nodes = data['house'].num_nodes
    idx = torch.randperm(num_nodes)
    split = int(0.8 * num_nodes)
    train_idx = idx[:split] # Entrenamiento
    test_idx = idx[split:] # Prueba

    y = data['house'].x[:, price_index]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

    # Evaluacion del modelo
    model.eval()
    with torch.no_grad():
        out = model(data)
        y_true = y[test_idx].cpu().numpy()
        y_pred = out[test_idx].cpu().numpy()
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

    return mse, mae, r2