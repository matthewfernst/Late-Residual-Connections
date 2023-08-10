import late_residual_connections as lrc
import yaml

if __name__ == "__main__":
    with open('experiment_vars.yml', 'r') as file:
        experiment_vars = yaml.safe_load(file)
    lrc.run(experiment_vars['width'],
            experiment_vars['depths'],
            experiment_vars['learning_rates'],
            experiment_vars['optimizers'],
            experiment_vars['epochs'])
