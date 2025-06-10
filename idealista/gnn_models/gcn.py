import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class GCN(torch.nn.Module):
    """
    Modelo de Red Neuronal Convolucional de Grafos (GCN) para regresion de nodos.
    Permite configurar el número de capas ocultas como hiperparámetro.
    """
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5):
        """
        Inicializa el modelo GCN con un número variable de capas de convolución y una tasa de dropout.

        Parámetros:
        in_channels: Número de características de entrada por nodo
        hidden_channels: Número de canales ocultos en las capas GCN
        num_layers: Número total de capas GCN (mínimo 2)
        dropout: Tasa de dropout para la regularización
        """
        super(GCN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        # Primera capa
        self.convs.append(GCNConv(in_channels, hidden_channels))
        # Capas ocultas intermedias (si las hay)
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        # Última capa
        self.convs.append(GCNConv(hidden_channels, 1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr.squeeze() if data.edge_attr is not None else None
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x.squeeze()

def train_node_regression(data, price_index=0, epochs=100, lr=0.01, hidden_channels=32, num_layers=2, dropout=0.5):
    """
    Entrena un modelo GCN para la regresión de nodos en un grafo, permitiendo configurar el número de capas.

    Parámetros:
    data: Objeto de datos de PyTorch Geometric que contiene las características de los nodos y las conexiones.
    price_index: Índice de la característica del precio en los nodos.
    epochs: Número de épocas para entrenar el modelo.
    lr: Tasa de aprendizaje para el optimizador.
    hidden_channels: Número de canales ocultos en las capas GCN.
    num_layers: Número total de capas GCN (mínimo 2).
    dropout: Tasa de dropout para la regularización.

    Returns:
    mse: Error cuadrático medio del modelo en el conjunto de prueba.
    mae: Error absoluto medio del modelo en el conjunto de prueba.
    r2: Coeficiente de determinación R^2 del modelo en el conjunto de prueba.
    """
    if data.num_nodes < 10:
        return float('nan'), float('nan'), float('nan')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(data.num_features, hidden_channels, num_layers, dropout).to(device)
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

    # Evaluación
    model.eval()
    with torch.no_grad():
        out = model(data)
        y_true = y[test_idx].cpu().numpy()
        y_pred = out[test_idx].cpu().numpy()
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
    
    return mse, mae, r2