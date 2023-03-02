import torch
import pygad
import os
import pickle
import yaml
import numpy as np
from copy import deepcopy

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
X_COORDINATES = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
EPS = 0.000001
# TODO: Argparse those values
PLOTTING = True
N_BEST_SOLUTIONS = 5

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


def inv_normalize(input, mean, std):
    mean_vals = [mean[1], mean[2], mean[3]]
    std_vals = [std[1], std[2], std[3]]
    return input*std_vals + mean_vals


def fitness_func(solution, sol_idx):
    """
    Genetic algorithm fitness function
    """
    global model, mean, std

    assert (len(solution) == 3)
    denorm_solution = inv_normalize(solution, mean, std)
    initial_temperature = denorm_solution[0]
    heat_input = denorm_solution[1]
    electrode_velocity = denorm_solution[2]

    T, R1, R2 = [], [], []
    for x_coord in X_COORDINATES:
        # NN model prediction
        input = np.array([[PLATE_THICKNESS, initial_temperature, heat_input,
                           electrode_velocity, x_coord, Y_COORDINATE, Z_COORDINATE]])
        input_scaled = normalize(input, mean, std).astype(np.float32)
        max_temperature_prediction = model.predict(input_scaled)
        T.append(max_temperature_prediction)

        # Constrain check for each prediction
        r1_val, r2_val = 0, 0

        if (max_temperature_prediction < K):
            r1_val = 1
        R1.append(r1_val)

        if (max_temperature_prediction > A):
            r2_val = 1
        R2.append(r2_val)

    # Calculate total cost
    cost_values = np.power(np.array(T) - (K+A)/2, 2) + \
        np.array(R1)*np.power(np.array(T) - K, 2) + \
        np.array(R2)*np.power(np.array(T) - A, 2)
    total_cost = np.sum(cost_values)

    # Fitness function of GA
    solution_fitness = 1.0 / (total_cost + EPS)
    return solution_fitness


def on_generation(ga_instance):
    """
    Callback for optimization info
    """
    print("Generation = {generation}".format(
        generation=ga_instance.generations_completed))
    print("Fitness = {fitness}".format(
        fitness=ga_instance.best_solution()[1]))
    print("--------------------------------------")

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
    gene_type = [float, float, float]
    ga_instance = pygad.GA(num_generations=ga_params["number_of_generations"],
                           num_parents_mating=ga_params["number_of_parents_mating"],
                           fitness_func=fitness_func,
                           sol_per_pop=ga_params["solutions_per_population"],
                           num_genes=ga_params["number_of_genes"],
                           gene_type=gene_type,
                           keep_elitism=ga_params["keep_elitism"],
                           init_range_low=ga_params["low_range"],
                           init_range_high=ga_params["high_range"],
                           parent_selection_type=ga_params["parent_selection_type"],
                           keep_parents=ga_params["keep_parents"],
                           crossover_type=ga_params["crossover_type"],
                           mutation_type=ga_params["mutation_type"],
                           mutation_percent_genes=ga_params["mutation_percent_genes"],
                           crossover_probability=ga_params["crossover_probability"],
                           on_generation=on_generation,
                           mutation_by_replacement=True,
                           random_mutation_min_val=ga_params["minimum_value_random_mutation"],
                           random_mutation_max_val=ga_params["maximum_value_random_mutation"],
                           save_best_solutions=True,
                           save_solutions=True,
                           random_seed=5
                           )
    ga_instance.run()

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    denormalized_solutions = inv_normalize(solution, mean, std)
    best_initial_temperature, best_heat_input, best_electrode_velocity = denormalized_solutions[
        0], denormalized_solutions[1], denormalized_solutions[2]
    print("Best solution genes: ")
    print(f"Initial temperature : {best_initial_temperature:.2f}")
    print(f"Heat Input : {best_heat_input:.2f}")
    print(f"Electrode Velocity : {best_electrode_velocity:.5f}")
    print("")

    # Return <N> best solutions of last generation
    all_solutions_len = len(ga_instance.solutions)
    step = ga_params["solutions_per_population"]
    last_generation_solutions = ga_instance.solutions[all_solutions_len -
                                                      step: all_solutions_len]
    last_generation_solutions_sorted = deepcopy(last_generation_solutions)
    last_generation_solutions_sorted.sort(
        key=lambda x: fitness_func(np.array(x), 0), reverse=True)
    n_best_last_generation_solutions = last_generation_solutions_sorted[:N_BEST_SOLUTIONS]
    n_best_last_generation_solutions_denorm = [inv_normalize(
        np.array(x), mean, std) for x in n_best_last_generation_solutions]
    print(
        f"Best {N_BEST_SOLUTIONS} solutions on last generation are: ")
    for sol in n_best_last_generation_solutions_denorm:
        initial_temperature, heat_input, electrode_velocity = sol[0], sol[1], sol[2]
        print(
            f"Initial temperature: {initial_temperature:.2f}, Heat Input: {heat_input:.2f}, Electrode Velocity: {electrode_velocity:.5f}")
    print("")

    # Neural network predictions, using `best_solution` along x axis
    nn_predictions = []
    for x_coord in X_COORDINATES:
        input = np.array([[PLATE_THICKNESS, best_initial_temperature, best_heat_input,
                           best_electrode_velocity, x_coord, Y_COORDINATE, Z_COORDINATE]])
        input_scaled = normalize(input, mean, std).astype(np.float32)
        max_temperature_prediction = model.predict(input_scaled)
        nn_predictions.append((x_coord, max_temperature_prediction[0][0]))
    print("Maximum temperatures predictions along x-axis, using best solution values: ")
    for x, pred in nn_predictions:
        print(
            f"x-axis coordinate: {x}, Maximum temperature prediction: {pred:.2f}")
    print("")

    # Neural network predictions, using top `N` best solutions along x axis
    best_n_nn_predictions = []
    for x_coord in X_COORDINATES:
        for sol in n_best_last_generation_solutions_denorm:
            initial_temperature, heat_input, electrode_velocity = sol[0], sol[1], sol[2]
            input = np.array([[PLATE_THICKNESS, initial_temperature, heat_input,
                               electrode_velocity, x_coord, Y_COORDINATE, Z_COORDINATE]])
            input_scaled = normalize(input, mean, std).astype(np.float32)
            max_temperature_prediction = model.predict(input_scaled)
            best_n_nn_predictions.append(
                (x_coord, initial_temperature, heat_input, electrode_velocity, max_temperature_prediction[0][0]))
    for x, t, h, e, pred in best_n_nn_predictions:
        print(
            f"x-axis coordinate: {x}, Initial temperature: {t:.2f}, Heat Input: {h:.2f}, Electrode Velocity: {e:.5f}, Maximum temperature prediction: {pred:.2f}")
    print("")

    # Plot Genetic Algorithm results
    if (PLOTTING):
        ga_instance.plot_fitness(
            title="GA Optimization Fitness", xlabel="Generations", ylabel="Fitness", linewidth=4)
        ga_instance.plot_genes(title="Genes", solutions="best")
