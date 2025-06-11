import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class GCN(torch.nn.Module):
    """
    Modelo GCN para regresion de nodos en grafos, con un numero variable de capas de convolucion.
    """
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        """
        La funcion inicializa el modelo GCN.

        Parametros:
        in_channels (int): Numero de caracteristicas de entrada por nodo.
        hidden_channels (int): Numero de canales ocultos en las capas GCN.
        num_layers (int): Numero total de capas GCN (minimo 2).
        dropout (float): Tasa de dropout para la regularizacion.

        Returns:
        Ninguno.

        Raises:
        ValueError: Si num_layers es menor que 2.
        TypeError: Si los argumentos no son del tipo esperado (int para in_channels, hidden_channels, num_layers; float para dropout).
        Exception: Si ocurre un error al inicializar las capas del modelo.
        """
        super(GCN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()

        # Primera capa
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # Capas ocultas intermedias (si hubieran)
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Ultima capa
        self.convs.append(GCNConv(hidden_channels, 1))

    def forward(self, data):
        """
        La funcion realiza el paso hacia adelante del modelo GCN.

        Parametros:
        data (torch_geometric.data.Data): Objeto de datos de PyTorch Geometric que contiene el grafo.

        Returns:
        x (torch.Tensor): Tensor de salida con las predicciones de precios por nodo.

        Raises:
        AttributeError: Si el objeto data no tiene los atributos esperados (x, edge_index, edge_attr).
        RuntimeError: Si ocurre un error durante el paso hacia adelante (por ejemplo, dimensiones incompatibles).
        Exception: Si ocurre cualquier otro error durante la propagacion hacia adelante.
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr.squeeze() if data.edge_attr is not None else None
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x.squeeze()

def train_node_regression(data, price_index, epochs, lr, hidden_channels, num_layers, dropout):

    """
    Entrena un modelo GCN para la regresion de precios de nodos en un grafo.

    Parametros:
    data (torch_geometric.data.Data): Objeto de datos de PyTorch Geometric que contiene el grafo.
    price_index (int): Indice de la caracteristica de precio en data.x.
    epochs (int): Numero de epocas para entrenar el modelo.
    lr (float): Tasa de aprendizaje para el optimizador.
    hidden_channels (int): Numero de canales ocultos en las capas GCN.
    num_layers (int): Numero total de capas GCN (minimo 2).
    dropout (float): Tasa de dropout para la regularizacion.

    Returns:
    mse (float): Error cuadratico medio en el conjunto de prueba.
    mae (float): Error absoluto medio en el conjunto de prueba.
    r2 (float): Coeficiente de determinacion R^2 en el conjunto de prueba.

    Raises:
    AttributeError: Si el objeto data no tiene los atributos esperados (x, edge_index, edge_attr).
    IndexError: Si price_index esta fuera de rango para data.x.
    ValueError: Si num_layers es menor que 2 o si los tamaños de los tensores no son compatibles.
    RuntimeError: Si ocurre un error durante el entrenamiento o la evaluacion (por ejemplo, problemas de CUDA o PyTorch).
    Exception: Si ocurre cualquier otro error durante el entrenamiento o evaluacion.
    """

    # No entrena si hay menos de 10 nodos
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