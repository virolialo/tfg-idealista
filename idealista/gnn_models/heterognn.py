import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, num_layers, dropout):
        """
        Modelo HeteroGNN para regresion de nodos en grafos heterogeneos.

        Parametros:
        metadata tuple(list[str], list[tuple]): Metadata del grafo heterogeneo.
        hidden_channels (int): Numero de canales ocultos para las capas GCN.
        num_layers (int): Numero total de capas GCN (minimo 2).
        dropout (float): Tasa de dropout para la regularizacion.

        Returns:
        Ninguno.

        Raises:
        ValueError: Si num_layers es menor que 2.
        TypeError: Si los argumentos no son del tipo esperado (metadata debe ser tuple, hidden_channels y num_layers int, dropout float).
        Exception: Si ocurre un error al inicializar las capas del modelo.
        """
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        # Construye una lista de HeteroConv para multiples capas
        self.convs = torch.nn.ModuleList()

        # Primera capa
        self.convs.append(HeteroConv({
            edge_type: GCNConv(-1, hidden_channels)
            for edge_type in metadata[1]
        }, aggr='sum'))

        # Capas ocultas intermedias (si hubieran)
        for _ in range(num_layers - 2):
            self.convs.append(HeteroConv({
                edge_type: GCNConv(hidden_channels, hidden_channels)
                for edge_type in metadata[1]
            }, aggr='sum'))

        # Ultima capa
        self.convs.append(HeteroConv({
            edge_type: GCNConv(hidden_channels, hidden_channels)
            for edge_type in metadata[1]
        }, aggr='sum'))
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        """
        La funcion realiza el paso hacia adelante del modelo HeteroGNN.

        Parametros:
        data (torch_geometric.data.HeteroData): HeteroData que contiene el grafo y las caracteristicas de los nodos.

        Returns:
        x (torch.Tensor): Tensor de salida con las predicciones de precios por nodo.

        Raises:
        AttributeError: Si el objeto data no tiene los atributos esperados (x_dict, edge_index_dict, etc.).
        KeyError: Si falta el tipo de nodo 'house' en data.x_dict.
        RuntimeError: Si ocurre un error durante el forward (por ejemplo, dimensiones incompatibles).
        Exception: Si ocurre cualquier otro error durante la propagacion hacia adelante.
        """
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1: # Aplica activacion y dropout en las capas intermedias
                for node_type in x_dict:
                    x_dict[node_type] = F.relu(x_dict[node_type])
                    x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout, training=self.training)
        x = x_dict['house']
        x = self.lin(x)
        return x.squeeze()

def train_node_regression_hetero(data, price_index, epochs, lr, hidden_channels, num_layers, dropout):
    """
    La funcion entrena un modelo HeteroGNN para la regresion de precios de nodos en un grafo heterogeneo.

    Parametros:
    data (torch_geometric.data.HeteroData): HeteroData que contiene el grafo y las caracteristicas de los nodos.
    price_index (int): Indice de la caracteristica del precio en los nodos.
    epochs (int): Numero de epocas para el entrenamiento.
    lr (float): Tasa de aprendizaje para el optimizador.
    hidden_channels (int): Numero de canales ocultos para las capas GCN.
    num_layers (int): Numero total de capas GCN (minimo 2).
    dropout (float): Tasa de dropout para la regularizacion.

    Returns:
    mse (float): Error cuadratico medio en el conjunto de prueba.
    mae (float): Error absoluto medio en el conjunto de prueba.
    r2 (float): Coeficiente de determinacion R^2 en el conjunto de prueba.

    Raises:
    AttributeError: Si el objeto data no tiene los atributos esperados (por ejemplo, 'house', 'x', etc.).
    IndexError: Si price_index esta fuera de rango para data['house'].x.
    ValueError: Si num_layers es menor que 2 o si los tamaños de los tensores no son compatibles.
    RuntimeError: Si ocurre un error durante el entrenamiento o la evaluacion (por ejemplo, problemas de CUDA o PyTorch).
    Exception: Si ocurre cualquier otro error durante el entrenamiento o evaluacion.
    """

    # No entrena si hay menos de 10 nodos
    if data['house'].num_nodes < 10:
        print(f"Saltando grafo con solo {data['house'].num_nodes} nodos.")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeteroGNN(data.metadata(), hidden_channels, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data = data.to(device)

    num_nodes = data['house'].num_nodes
    idx = torch.randperm(num_nodes)
    split = int(0.8 * num_nodes)
    train_idx = idx[:split]
    test_idx = idx[split:]

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