import torch
import pygad
import os
import pickle
import yaml
import numpy as np

# TODO: Add docstrings

# *==== Define Constants ====*
WEIGHTS_PATH = "/nn_weights/MaximumTemperatureNN_Model_Parameters.pt"
PREPROCESS_VALUES_PATH = "/nn_weights/MaximumTemperatureNN_training_preprocess_values.pickle"
DATA_PATH = "/data/nn1/"
GA_PARAMETERS_PATH = "/data/ga/parameters.yaml"

K = 750
A = 1400
PLATE_THICKNESS = 0.005
Y_COORDINATE = 0.02 + 0.004
Z_COORDINATE = 0.0
X_COORDINATES = [0.0, 0.015, 0.035, 0.05]
EPS = 0.000001

# *==== Class Declaration ====*


class MaxTemperatureNN(torch.nn.Module):
    """
    Neural Network Model for maximum temperature regression
    of a welding process
    """

    def __init__(self, input=7, output=1):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 32),
            torch.nn.Sigmoid(),
            torch.nn.Linear(32, output)
        )

    def forward(self, x):
        return self.layers(x)

    def predict(self, x):
        with torch.no_grad():
            x = torch.tensor(x)
            prediction = self.forward(x).detach().cpu().numpy()
        return prediction


# *==== Method Definitions ====*

def normalize(input, mean, std):
    return (input - mean) / std


def fitness_func(solution, sol_idx):
    """
    Genetic algorithm fitness function
    """
    global model, mean, std

    assert (len(solution) == 3)
    initial_temperature = solution[0]
    heat_input = solution[1]
    electrode_velocity = solution[2]

    F, R1, R2 = [], [], []
    for x in X_COORDINATES:
        # NN model prediction
        input = np.array([[PLATE_THICKNESS, initial_temperature, heat_input,
                           electrode_velocity, x, Y_COORDINATE, Z_COORDINATE]])
        input_scaled = normalize(input, mean, std).astype(np.float32)
        max_temperature_prediction = model.predict(input_scaled)

        f_val = max_temperature_prediction - (K+A)/2
        F.append(f_val)

        # Constrain check for each prediction
        r1_val, r2_val = 0, 0

        if (max_temperature_prediction < K):
            r1_val = 1
        R1.append(r1_val)

        if (max_temperature_prediction > A):
            r2_val = 1
        R2.append(r2_val)

    # Calculate total cost
    cost_values = np.power(np.array(F) - (K+A)/2, 2) + \
        np.array(R1)*np.power(np.array(F) - K, 2) + \
        np.array(R2)*np.power(np.array(F) - A, 2)
    total_cost = np.sum(cost_values)

    # Fitness function of GA
    solution_fitness = 1.0 / (total_cost + EPS)
    return solution_fitness


def generation_callback(ga_instance):
    """
    Callback for optimization info
    """
    print("Generation = {generation}".format(
        generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(
        fitness=ga_instance.best_solution()[1]))

    return


def load_values():
    """
    Helper method that reads cached values about neural network definition
    """
    home_folder = os.getcwd()
    model = MaxTemperatureNN(input=7, output=1)
    model.load_state_dict(torch.load(
        home_folder+WEIGHTS_PATH, map_location=torch.device("cpu")))
    model.to("cpu")

    # Read and save preprocessing values
    preprocess_filename = home_folder + PREPROCESS_VALUES_PATH
    with open(preprocess_filename, "rb") as pre_fname:
        train_values_data = pickle.load(pre_fname)
    mean = train_values_data["mean"]
    std = train_values_data["std"]

    return model, mean, std


if (__name__ == "__main__"):

    # Obtain trained neural network
    model, mean, std = load_values()

    # Parse GA optimization parameters
    ga_param_filename = os.getcwd() + GA_PARAMETERS_PATH
    with open(ga_param_filename) as param_file:
        ga_params = yaml.load(param_file, Loader=yaml.FullLoader)

    # Create Genetic algorithm
    gene_type = [int, int, float]
    gene_space = [{'low': 0, 'high': 500},  # `InitialTemperature` parameter
                  {'low': 100, 'high': 2000},  # For the P parameter
                  {'low': 0.0001, 'high': 1.0}]  # For the ElectrodeVelocity parameter
    ga_instance = pygad.GA(num_generations=ga_params["number_of_generations"],
                           num_parents_mating=ga_params["number_of_parents_mating"],
                           fitness_func=fitness_func,
                           sol_per_pop=ga_params["solutions_per_population"],
                           num_genes=ga_params["number_of_genes"],
                           gene_type=gene_type,
                           gene_space=gene_space,
                           parent_selection_type=ga_params["parent_selection_type"],
                           keep_parents=ga_params["keep_parents"],
                           crossover_type=ga_params["crossover_type"],
                           mutation_type=ga_params["mutation_type"],
                           mutation_percent_genes=ga_params["mutation_percent_genes"],
                           callback_generation=generation_callback
                           )
    ga_instance.run()

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(
        solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(
        solution_idx=solution_idx))

    # Plot Genetic Algorithm results
    ga_instance.plot_fitness(
        title="Iteration vs. Fitness", linewidth=4)
