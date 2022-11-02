## Late Residual Connections

This repository is a cleaned version of my research on the vanishing gradient problem with dead ReLUs. My research suggests an approach to combat dead ReLUs with residual connections called Late Residual Connections. These connections are added directly to the output of a neural network from each hidden layer. Please feel free to use the code and add on to the research! I can be reached [here](mailto:matthew.f.ernst@gmail.com) for any questions.

### Example of A Late Residual Connection Neural Network

Below is an example of an abstracted view of traditional versus late residual neural networks. Each hidden layer has a late residual connection the output. To simplify, this connection allows for moregradient flow to each neuron in the network. My full thesis can be read [here](thesis.pdf).

![late-residual](Images/lrn.png)


## Setup and How to Run

[run_experiment_script.py](Code/run_experiment_script.py) is a script that can be used to run the experiments.
Below explains how to create an environment and run the script.

### Creating Environment

You will need a Python version of 3.8 or higher to run the code in this repository. Run the following commands below to create your environment and install the required packages.

```bash
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```


### Running the Script

To run the script, run [run_experiment_script.py](Code/run_experiment_script.py). This script has argument parasing to change the following parameters
    
    - "--width": The widths of the networks.
    - "--depths": The depths of the networks.
    - "--lr": The learning rates of the networks.
    - "--optimizers": The optimizers of the networks.
    - "--epochs": The number of epochs to train each network.

To see more information run 
```bash 
run_experiment_script.py --help
```



