import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class GCN(torch.nn.Module):
    """
    Modelo de Red Neuronal Convolucional de Grafos (GCN) para regresion de nodos.
    Este modelo toma caracteristicas de nodos y conexiones de un grafo y predice un valor continuo para cada nodo.
    """
    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        """
        La funcion inicializa el modelo GCN con dos capas de convolucion y una tasa de dropout.

        Parametros:
        in_channels: Numero de caracteristicas de entrada por nodo
        hidden_channels: Numero de canales ocultos en la capa GCN
        dropout: Tasa de dropout para la regularizacion

        Returns:
        Ninguno
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, data):
        """
        La funcion de avance del modelo GCN.
        Toma un objeto de datos que contiene las caracteristicas de los nodos y las conexiones del grafo,
        y devuelve las predicciones de regresion para cada nodo.

        Parametros:
        data: Objeto de datos de PyTorch Geometric que contiene las caracteristicas de los nodos y las conexiones.

        Returns:
        Tensor: Predicciones de regresion para cada nodo en el grafo.
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr.squeeze() if data.edge_attr is not None else None
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x.squeeze()

def train_node_regression(data, price_index=0, epochs=100, lr=0.01, hidden_channels=32, dropout=0.5):
    """
    La funcion entrena un modelo GCN para la regresion de nodos en un grafo.

    Parametros:
    data: Objeto de datos de PyTorch Geometric que contiene las caracteristicas de los nodos y las conexiones.
    price_index: Indice de la caracteristica del precio en los nodos.
    epochs: Numero de epocas para entrenar el modelo.
    lr: Tasa de aprendizaje para el optimizador.
    hidden_channels: Numero de canales ocultos en la capa GCN.
    dropout: Tasa de dropout para la regularizacion.

    Returns:
    mse: Error cuadratico medio del modelo en el conjunto de prueba.
    mae: Error absoluto medio del modelo en el conjunto de prueba.
    r2: Coeficiente de determinacion R^2 del modelo en el conjunto de prueba.
    """
    # Modelos con menos de 10 nodos se descartan
    if data.num_nodes < 10:
        return float('nan'), float('nan'), float('nan')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(data.num_features, hidden_channels, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data = data.to(device)

    num_nodes = data.num_nodes
    idx = torch.randperm(num_nodes)
    split = int(0.8 * num_nodes)
    train_idx = idx[:split]
    test_idx = idx[split:]

    y = data.x[:, price_index]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

    # Evaluacion
    model.eval()
    with torch.no_grad():
        out = model(data)
        y_true = y[test_idx].cpu().numpy()
        y_pred = out[test_idx].cpu().numpy()
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
    
    return mse, mae, r2
