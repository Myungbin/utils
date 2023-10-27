import torch
import torch.nn as nn


class TabularModel(nn.Module):
    def __init__(self, input_dims, start_neurons):
        super(TabularModel, self).__init__()

        # Embedding layers
        self.embeddings = nn.ModuleList([nn.Embedding(dim, start_neurons) for dim in input_dims[:-1]])
        self.linear_embedding = nn.Linear(1, start_neurons)

        # Main layers
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(5)])
        self.gates = nn.ModuleList(
            [nn.Linear(start_neurons * len(input_dims), start_neurons * len(input_dims)) for _ in range(5)])
        self.linear_layers = nn.ModuleList(
            [nn.Linear(start_neurons * len(input_dims) * 2, start_neurons * len(input_dims)) for _ in range(5)])

        # Output layers
        self.output_layers = nn.ModuleList([nn.Linear(start_neurons * len(input_dims), 1) for _ in range(10)])

    def forward(self, x):
        embeddings = [self.embeddings[i](x[:, i]) for i in range(len(input_dims[:-1]))]
        embeddings.append(self.linear_embedding(x[:, -1].float().unsqueeze(-1)))

        concatenated_inputs = torch.cat(embeddings, dim=1)

        for dropout, gate, dense in zip(self.dropouts, self.gates, self.linear_layers):
            dropped_input = dropout(concatenated_inputs)
            gate_output = torch.sigmoid(gate(dropped_input))
            gated_input = concatenated_inputs * gate_output
            concat_input = torch.cat([concatenated_inputs, gated_input], dim=1)
            concatenated_inputs = concatenated_inputs + dense(concat_input)

        outputs = [layer(concatenated_inputs) for layer in self.output_layers]
        output = torch.mean(torch.cat(outputs, dim=1), dim=1)
        return output
