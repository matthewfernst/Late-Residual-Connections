from typing import List, Tuple

import torch
import numpy as np

class ResidualBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.residual_connection = torch.nn.Identity()
        
    def forward (self, x):
        return self.residual_connection(x)


class NNet(torch.nn.Module):
    def __init__(self, n_inputs: int, n_hiddens_list: List[int], n_outputs: int, optimizer: str,
                 isResiduallyConnected: bool = False, device: str = None):
        super().__init__()

        if device is None:
            self.device = torch.device("cpu")
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            
        self.error_trace = []
        self.optimizer_selected = optimizer
        self.isResiduallyConnected = isResiduallyConnected
        
        # Set self.n_hiddens_per_layer to [] if argument is 0, [], or [0]
        if n_hiddens_list == 0 or n_hiddens_list == [] or n_hiddens_list == [0]:
            layers = torch.nn.Linear(n_inputs, n_outputs)

        elif self.isResiduallyConnected:
            self.input_layer = torch.nn.Linear(n_inputs, n_hiddens_list[0])
            self.input_relu = torch.nn.ReLU()
            
            layers = []
            self.residual_layers = []
            self.regular_layers = []
            
            for nh in n_hiddens_list[:-1]:
                self.regular_layers.append(torch.nn.Linear(n_inputs, nh))
                layers.append(self.regular_layers[-1])
                self.regular_layers.append(torch.nn.ReLU())
                layers.append(self.regular_layers[-1])
                n_inputs = nh
                self.residual_layers.append(self.get_residual_layer())
                layers.append(self.residual_layers[-1])
                
                
            layers.append(torch.nn.Linear(n_inputs, n_hiddens_list[-1]))
            layers.append(torch.nn.ReLU())
            self.output_layer = torch.nn.Linear(np.sum(n_hiddens_list), n_outputs)
            layers.append(self.output_layer)
            
        else:
            layers = []
            for nh in n_hiddens_list:
                layers.append(torch.nn.Linear(n_inputs, nh))
                layers.append(torch.nn.ReLU())
                n_inputs = nh
                    
            layers.append(torch.nn.Linear(n_inputs, n_outputs))

        self.model = torch.nn.Sequential(*layers)
        self.model.to(self.device)

        self.x_means = None
        self.x_stds = None
        self.t_means = None
        self.t_stds = None

        
    def get_residual_layer(self) -> torch.nn.Module:
        """
        Helper function to create a residual layer.
        :return: A residual layer.
        """
        layers = []
        layers.append(ResidualBlock())
        return torch.nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        :param x: An MxN matrix of M training cases each with N inputs.
        :return: An MxK matrix of M predicted values for each of K outputs.
        """

        residual_outs = []
        self.layer_outputs = []
        
        for layer in self.model[:-1]:
            # For Regular Layers          
            if isinstance(layer, torch.nn.Linear):
                x = layer(x)
            elif isinstance(layer, torch.nn.ReLU):
                x = layer(x)
                self.layer_outputs.append(x.cpu().detach().numpy())

            # For Residual Layers
            elif isinstance(layer, torch.nn.Sequential):
                residual_outs.append(x)
            else:
                raise Exception(f"{layer} is not permitted.")
        
        # Concat all residual outs and normal out and push through output layer
        if self.isResiduallyConnected:
            residual_outs.append(x)
            x = self.output_layer(torch.cat(tuple(residual_outs),dim=1))
        else:
            x = self.model[-1](x)

        return x
        
    def train(self, x: torch.tensor, t: torch.tensor, epochs: int, learning_rate: float, training_style: str,
              verbose=True) -> None:
        """
        Train the neural network using the given inputs, targets, and hyperparameters.
        :param x: The input data
        :param t: The target data
        :param epochs: The number of epochs to train for
        :param learning_rate: The learning rate
        :param training_style: The training style to use
        """

        # Set data matrices to torch.tensors if not already.
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        if not isinstance(t, torch.Tensor):
            t = torch.from_numpy(t).float()
            
        # Calculate standardization parameters if not already calculated
        if self.x_means is None:
            self.x_means = x.mean(0)
            self.x_stds = x.std(0)
            self.x_stds[self.x_stds == 0] = 1
            self.t_means = t.mean(0)
            self.t_stds = t.std(0)
            self.t_stds[self.t_stds == 0] = 1

        # Move data over to compatible device if available 
        if self.device != torch.device("cpu"):
            self.x_means = self.x_means.to(self.device)
            self.x_stds = self.x_stds.to(self.device)
            self.t_means = self.t_means.to(self.device)
            self.t_stds = self.t_stds.to(self.device)

            x = x.to(self.device)
            t = t.to(self.device)
            self = self.to(self.device)

            
        # Standardize inputs and targets
        x = (x - self.x_means) / self.x_stds
        t = (t - self.t_means) / self.t_stds
        
        
        # Set optimizer to Adam / SGD and loss functions to MSELoss
        optimizer = None
        if self.optimizer_selected == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif self.optimizer_selected == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        else:
            raise ValueError("Only \"Adam\" or \"SGD\" Optimizers are currently supported. "
                             f"Got {self.optimizer_selected}")
        criterion = torch.nn.MSELoss()

        unstndErr = lambda err:(torch.sqrt(err) * self.t_stds)[0]

        for epoch in range(epochs):
            
            # Compute Prediction and loss
            if training_style == "Batch":
                y = self.forward(x)
                loss = criterion(t, y)
            elif training_style == "Iterative":
                y = self.forward(x[epoch % len(x)].reshape(-1,1))
                loss = criterion(t[epoch % len(t)].reshape(-1,1), y)
            else:
                raise ValueError(f"Training style must be \"Batch\" or \"Iterative\". Got {training_style}")
            
            # Backpropigation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
          
            self.error_trace.append(unstndErr(loss))
            
            if verbose and ((epoch+1 == epochs) or np.mod(epoch+1 , (epochs // 10)) == 0):
                print(f"Epoch {epoch + 1}: RMSE {self.error_trace[-1]:.3f}")
            

    def use(self, x: torch.tensor) -> torch.tensor:
        """
        Use the trained model to predict target values.
        :param x: An MxN matrix of M training cases each with N inputs.
        :return: An MxK matrix of M predicted values for each of K outputs.
        """
       # Set input matrix to torch.tensors if not already.
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()

        # Standardize x
        if self.device != torch.device("cpu"):
            x = x.to(self.device)
        x = (x - self.x_means) / self.x_stds
        
        # Do forward pass and unstandardize resulting output. Assign to variable y.
        y = self.forward(x)
        y = (y * self.t_stds) + self.t_means

        # Return output y after detaching from computation graph and converting to numpy
        if self.device != torch.device("cpu"):
            y = y.cpu()

        return y.detach().numpy()


    def dead_neurons(self) -> Tuple[int, int]:
        """
        Returns the number of dead neurons and dead layers for a given network
        :return: Tuple[int, int]
        """
        dead_neurons = 0
        all_layer_dead = []
        dead_layers = 0
        
        for layer in self.layer_outputs:
            for neuron in layer.T:
                if np.all(neuron == 0):
                    dead_neurons += 1
                    all_layer_dead.append(True)
            if np.all(all_layer_dead):
                dead_layers += 1
            all_layer_dead = []
        return dead_neurons, dead_layers
