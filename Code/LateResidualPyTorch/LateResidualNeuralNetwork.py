from logging import exception
import torch
import numpy as np

class ResidualBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.residual_connection = torch.nn.Identity()
        
    def forward (self, x):
        return self.residual_connection(x)


class NNet(torch.nn.Module):
    
    def __init__(self, n_inputs, n_hiddens_list, n_outputs, optimizer, isResiduallyConnected=False):
        super().__init__()  # call parent class (torch.nn.Module) constructor
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

        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None

        
    def get_residual_layer(self):
        layers = []
        layers.append(ResidualBlock())
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):

        residual_outs = []
        self.layer_outputs = []
        
        for layer in self.model[:-1]:
            # For Regular Layers          
            if isinstance(layer, torch.nn.Linear):
                x = layer(x)
            elif isinstance(layer, torch.nn.ReLU):
                x = layer(x)
                self.layer_outputs.append(x.detach().numpy())

            # For Residual Layers
            elif isinstance(layer, torch.nn.Sequential):
                residual_outs.append(x)
            else:
                raise Exception(f'{layer} is not permitted.')
        
        # Concat all residual outs and normal out
        # push through to the outputlayer
        if self.isResiduallyConnected:
            residual_outs.append(x)
            x = self.output_layer(torch.cat(tuple(residual_outs),dim=1))
        else:
            x = self.model[-1](x)

        return x
        

    def train(self, X, T, epochs, learning_rate, training_style, verbose=True):

        # Set data matrices to torch.tensors if not already.
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        if not isinstance(T, torch.Tensor):
            T = torch.from_numpy(T).float()
            
        # Calculate standardization parameters if not already calculated
        if self.Xmeans is None:
            self.Xmeans = X.mean(0)
            self.Xstds = X.std(0)
            self.Xstds[self.Xstds == 0] = 1
            self.Tmeans = T.mean(0)
            self.Tstds = T.std(0)
            self.Tstds[self.Tstds == 0] = 1

            
        # Standardize inputs and targets
        X = (X - self.Xmeans) / self.Xstds
        T = (T - self.Tmeans) / self.Tstds
        
        
        # Set optimizer to Adam / SGD and loss functions to MSELoss
        optimizer = None
        if self.optimizer_selected == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif self.optimizer_selected == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        else:
            raise Exception(f'Must select \'Adam\' or \'SGD\' Optimizer. Got {self.optimizer_selected}')
        criterion = torch.nn.MSELoss()

        unstndErr = lambda err:(torch.sqrt(err) * self.Tstds)[0]

        for epoch in range(epochs):
            
            # Compute Prediction and loss
            if training_style == 'Batch':
                Y = self.forward(X)
                loss = criterion(T, Y)
            elif training_style == 'Iterative':
                Y = self.forward(X[epoch % len(X)].reshape(-1,1))
                loss = criterion(T[epoch % len(T)].reshape(-1,1), Y)
            else:
                raise exception(f'Training style must be \'Batch\' or \'Iterative\'. Got {training_style}')
            
            # Backpropigation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
          
            self.error_trace.append(unstndErr(loss))
            
            if verbose and ((epoch+1 == epochs) or np.mod(epoch+1 , (epochs // 10)) == 0):
                print(f'Epoch {epoch + 1}: RMSE {self.error_trace[-1]:.3f}')
            

    def use(self, X):
 
       # Set input matrix to torch.tensors if not already.
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()

        # Standardize X
        X = (X - self.Xmeans) / self.Xstds
        
        # Do forward pass and unstandardize resulting output. Assign to variable Y.
        Y = self.forward(X)
        Y = (Y * self.Tstds) + self.Tmeans
        # Return output Y after detaching from computation graph and converting to numpy

        return Y.detach().numpy()


    def dead_neurons(self):
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