#from solvers.SolverRGBD_QuatLoss import *
import sys
sys.dont_write_bytecode = True
from models.Model_8n import *
from solvers.Solver_nrm_cm import *
from solvers.Solver_reim_cm import *
from solvers.Solver_nrm import *
from solvers.solver_nvalid import *

# This experiment works with the bunny model 128 x 128 pixels.
# The model color is adapted (chromatic adaptation) and noise was applied with sigma = 0.15
def exs(name):
    ex_test_valid = {}
    ex_test_valid["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex_test_valid["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex_test_valid["solver"] = solver_nvalid
    ex_test_valid["model"] = Model_8n
    ex_test_valid["learning_rate"] = 0.1
    ex_test_valid["batch_size"] = 1
    ex_test_valid["num_iterations"] = 100
    ex_test_valid["debug_output"] = True
    ex_test_valid["log_path"] = "./log_real_128/ex_test/"
    ex_test_valid["log_file"] = "plx64"
    ex_test_valid["train"] = False
    ex_test_valid["eval"] = True
    ex_test_valid["test"] = False
    ex_test_valid["proof"] = False
    ex_test_valid["restore_file"] = "plx64-10.meta"  # keep empty to not restore a model.
    ex_test_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex_test_valid["plot_title"] = "plx128_overfit: ex_test"
    ex_test_valid["label"] = "Experiment.py"
    if name == "ex_test_valid":
        name = ex_test_valid

    ex_test = {}
    ex_test["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex_test["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex_test["solver"] = Solver_reim_cm
    ex_test["model"] = Model_8n
    ex_test["batch_size"] = 128
    ex_test["learning_rate"] = 0.001
    ex_test["num_iterations"] = 10
    ex_test["debug_output"] = True
    ex_test["log_path"] = "./log_real_128/ex_test/"
    ex_test["log_file"] = "plx64"
    ex_test["train"] = True
    ex_test["eval"] = False
    ex_test["test"] = False
    ex_test["proof"] = False
    ex_test["stage2"] = False
    ex_test["cont"] = False
    ex_test["drp_cnv"] = 1.0
    ex_test["drp_pose"] = 1.0
    ex_test["restore_file"] = ""  # keep empty to not restore a model.
    ex_test["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex_test["plot_title"] = "plx128_overfit: ex_test"
    ex_test["label"] = "Experiment.py"
    if name == "ex_test":
        name = ex_test

    ex_test_1_valid = {}
    ex_test_1_valid["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex_test_1_valid["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex_test_1_valid["solver"] = solver_nvalid
    ex_test_1_valid["model"] = Model_8n
    ex_test_1_valid["learning_rate"] = 0.003
    ex_test_1_valid["batch_size"] = 5
    ex_test_1_valid["num_iterations"] = 100
    ex_test_1_valid["debug_output"] = True
    ex_test_1_valid["log_path"] = "./log_real_128/ex_test_1/"
    ex_test_1_valid["log_file"] = "plx64"
    ex_test_1_valid["train"] = False
    ex_test_1_valid["eval"] = True
    ex_test_1_valid["test"] = False
    ex_test_1_valid["proof"] = True
    ex_test_1_valid["restore_file"] = "plx64-100.meta"  # keep empty to not restore a model.
    ex_test_1_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex_test_1_valid["plot_title"] = "plx128_overfit: ex_test_1"
    ex_test_1_valid["label"] = "Experiment.py"
    if name == "ex_test_1_valid":
        name = ex_test_1_valid

    ex_test_1 = {}
    ex_test_1["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex_test_1["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex_test_1["solver"] = Solver_nrm
    ex_test_1["model"] = Model_8n
    ex_test_1["batch_size"] = 5
    ex_test_1["learning_rate"] = 0.009
    ex_test_1["num_iterations"] = 100
    ex_test_1["debug_output"] = True
    ex_test_1["log_path"] = "./log_real_128/ex_test_1/"
    ex_test_1["log_file"] = "plx64"
    ex_test_1["train"] = True
    ex_test_1["eval"] = False
    ex_test_1["test"] = False
    ex_test_1["proof"] = True
    ex_test_1["stage2"] = False
    ex_test_1["cont"] = False
    ex_test_1["drp_cnv"] = 1.0
    ex_test_1["drp_pose"] = 1.0
    ex_test_1["restore_file"] = "plx64-100.meta"  # keep empty to not restore a model.
    ex_test_1["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex_test_1["plot_title"] = "plx128_overfit: ex_test_1"
    ex_test_1["label"] = "Experiment.py"
    if name == "ex_test_1":
        name = ex_test_1

    ex1n_valid = {}
    ex1n_valid["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex1n_valid["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex1n_valid["solver"] = solver_nvalid
    ex1n_valid["model"] = Model_8n
    ex1n_valid["learning_rate"] = 0.001
    ex1n_valid["num_iterations"] = 100
    ex1n_valid["debug_output"] = True
    ex1n_valid["log_path"] = "./log_real_128/ex1n/"
    ex1n_valid["log_file"] = "plx64"
    ex1n_valid["train"] = False
    ex1n_valid["eval"] = True
    ex1n_valid["test"] = False
    ex1n_valid["proof"] = False
    ex1n_valid["restore_file"] = "plx64-200.meta"  # keep empty to not restore a model.
    ex1n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex1n_valid["plot_title"] = "plx128_overfit: ex1n"
    ex1n_valid["label"] = "Experiment.py"
    if name == "ex1n_valid":
        name = ex1n_valid

    ex1n = {}
    ex1n["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex1n["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex1n["solver"] = Solver_nrm_cm
    ex1n["model"] = Model_8n
    ex1n["learning_rate"] = 0.001
    ex1n["num_iterations"] = 100
    ex1n["debug_output"] = True
    ex1n["log_path"] = "./log_real_128/ex1n/"
    ex1n["log_file"] = "plx64"
    ex1n["train"] = True
    ex1n["eval"] = False
    ex1n["test"] = False
    ex1n["proof"] = False
    ex1n["stage2"] = False
    ex1n["cont"] = False
    ex1n["drp_cnv"] = 0.95
    ex1n["drp_pose"] = 0.8
    ex1n["restore_file"] = "plx64-100.meta"  # keep empty to not restore a model.
    ex1n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex1n["plot_title"] = "plx128_overfit: ex1n"
    ex1n["label"] = "Experiment.py"
    if name == "ex1n":
        name = ex1n

    ex3n_valid = {}
    ex3n_valid["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex3n_valid["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex3n_valid["solver"] = solver_nvalid
    ex3n_valid["model"] = Model_8n
    ex3n_valid["learning_rate"] = 0.001
    ex3n_valid["num_iterations"] = 100
    ex3n_valid["debug_output"] = True
    ex3n_valid["log_path"] = "./log_real_128/ex1n/ex3n/"
    ex3n_valid["log_file"] = "plx64"
    ex3n_valid["train"] = False
    ex3n_valid["eval"] = True
    ex3n_valid["test"] = False
    ex3n_valid["proof"] = False
    ex3n_valid["restore_file"] = "plx64-200.meta"  # keep empty to not restore a model.
    ex3n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex3n_valid["plot_title"] = "plx128_overfit: ex3n"
    ex3n_valid["label"] = "Experiment.py"
    if name == "ex3n_valid":
        name = ex3n_valid

    ex3n = {}
    ex3n["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex3n["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex3n["solver"] = Solver_nrm_cm
    ex3n["model"] = Model_8n
    ex3n["learning_rate"] = 0.001
    ex3n["num_iterations"] = 100
    ex3n["debug_output"] = True
    ex3n["log_path"] = "./log_real_128/ex1n/ex3n/"
    ex3n["log_file"] = "plx64"
    ex3n["train"] = True
    ex3n["eval"] = False
    ex3n["test"] = False
    ex3n["proof"] = False
    ex3n["stage2"] = True
    ex3n["cont"] = False
    ex3n["drp_cnv"] = 0.8
    ex3n["drp_pose"] = 0.8
    ex3n["restore_file"] = "plx64-100.meta"  # keep empty to not restore a model.
    ex3n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex3n["plot_title"] = "plx128_overfit: ex3n"
    ex3n["label"] = "Experiment.py"
    if name == "ex3n":
        name = ex3n

    ex4n_valid = {}
    ex4n_valid["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex4n_valid["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex4n_valid["solver"] = solver_nvalid
    ex4n_valid["model"] = Model_8n
    ex4n_valid["learning_rate"] = 0.001
    ex4n_valid["num_iterations"] = 100
    ex4n_valid["debug_output"] = True
    ex4n_valid["log_path"] = "./log_real_128/ex4n/"
    ex4n_valid["log_file"] = "plx64"
    ex4n_valid["train"] = False
    ex4n_valid["eval"] = True
    ex4n_valid["test"] = False
    ex4n_valid["proof"] = False
    ex4n_valid["restore_file"] = "plx64-200.meta"  # keep empty to not restore a model.
    ex4n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex4n_valid["plot_title"] = "plx128_overfit: ex4n"
    ex4n_valid["label"] = "Experiment.py"
    if name == "ex4n_valid":
        name = ex4n_valid

    ex4n = {}
    ex4n["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex4n["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex4n["solver"] = Solver_nrm_cm
    ex4n["model"] = Model_8n
    ex4n["learning_rate"] = 0.003
    ex4n["num_iterations"] = 100
    ex4n["debug_output"] = True
    ex4n["log_path"] = "./log_real_128/ex4n/"
    ex4n["log_file"] = "plx64"
    ex4n["train"] = True
    ex4n["eval"] = False
    ex4n["test"] = False
    ex4n["proof"] = False
    ex4n["stage2"] = False
    ex4n["cont"] = False
    ex4n["drp_cnv"] = 0.95
    ex4n["drp_pose"] = 0.95
    ex4n["restore_file"] = "plx64-100.meta"  # keep empty to not restore a model.
    ex4n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex4n["plot_title"] = "plx128_overfit: ex4n"
    ex4n["label"] = "Experiment.py"
    if name == "ex4n":
        name = ex4n

    ex5n_valid = {}
    ex5n_valid["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex5n_valid["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex5n_valid["solver"] = solver_nvalid
    ex5n_valid["model"] = Model_8n
    ex5n_valid["learning_rate"] = 0.001
    ex5n_valid["num_iterations"] = 100
    ex5n_valid["debug_output"] = True
    ex5n_valid["log_path"] = "./log_real_128/ex4n/ex5n/"
    ex5n_valid["log_file"] = "plx64"
    ex5n_valid["train"] = False
    ex5n_valid["eval"] = True
    ex5n_valid["test"] = False
    ex5n_valid["proof"] = False
    ex5n_valid["restore_file"] = "plx64-300.meta"  # keep empty to not restore a model.
    ex5n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex5n_valid["plot_title"] = "plx128_overfit: ex5n"
    ex5n_valid["label"] = "Experiment.py"
    if name == "ex5n_valid":
        name = ex5n_valid

    ex5n = {}
    ex5n["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex5n["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex5n["solver"] = Solver_nrm_cm
    ex5n["model"] = Model_8n
    ex5n["learning_rate"] = 0.001
    ex5n["num_iterations"] = 100
    ex5n["debug_output"] = True
    ex5n["log_path"] = "./log_real_128/ex4n/ex5n/"
    ex5n["log_file"] = "plx64"
    ex5n["train"] = True
    ex5n["eval"] = False
    ex5n["test"] = False
    ex5n["proof"] = False
    ex5n["stage2"] = True
    ex5n["cont"] = False
    ex5n["drp_cnv"] = 0.9
    ex5n["drp_pose"] = 0.9
    ex5n["restore_file"] = "plx64-200.meta"  # keep empty to not restore a model.
    ex5n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex5n["plot_title"] = "plx128_overfit: ex5n"
    ex5n["label"] = "Experiment.py"
    if name == "ex5n":
        name = ex5n

    ex6n_valid = {}
    ex6n_valid["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex6n_valid["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex6n_valid["solver"] = solver_nvalid
    ex6n_valid["model"] = Model_8n
    ex6n_valid["learning_rate"] = 0.001
    ex6n_valid["num_iterations"] = 100
    ex6n_valid["debug_output"] = True
    ex6n_valid["log_path"] = "./log_real_128/ex6n/"
    ex6n_valid["log_file"] = "plx64"
    ex6n_valid["train"] = False
    ex6n_valid["eval"] = True
    ex6n_valid["test"] = False
    ex6n_valid["proof"] = False
    ex6n_valid["restore_file"] = "plx64-200.meta"  # keep empty to not restore a model.
    ex6n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex6n_valid["plot_title"] = "plx128_overfit: ex6n"
    ex6n_valid["label"] = "Experiment.py"
    if name == "ex6n_valid":
        name = ex6n_valid

    ex6n = {}
    ex6n["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex6n["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex6n["solver"] = Solver_nrm_cm
    ex6n["model"] = Model_8n
    ex6n["learning_rate"] = 0.001
    ex6n["num_iterations"] = 100
    ex6n["debug_output"] = True
    ex6n["log_path"] = "./log_real_128/ex6n/"
    ex6n["log_file"] = "plx64"
    ex6n["train"] = True
    ex6n["eval"] = False
    ex6n["test"] = False
    ex6n["proof"] = False
    ex6n["stage2"] = False
    ex6n["only_r"] = False
    ex6n["cont"] = False
    ex6n["drp_cnv"] = 0.85
    ex6n["drp_pose"] = 0.9
    ex6n["restore_file"] = "plx64-100.meta"  # keep empty to not restore a model.
    ex6n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex6n["plot_title"] = "plx128_overfit: ex6n"
    ex6n["label"] = "Experiment.py"
    if name == "ex6n":
        name = ex6n

    ex7n_valid = {}
    ex7n_valid["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex7n_valid["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex7n_valid["solver"] = solver_nvalid
    ex7n_valid["model"] = Model_8n
    ex7n_valid["learning_rate"] = 0.001
    ex7n_valid["num_iterations"] = 100
    ex7n_valid["debug_output"] = True
    ex7n_valid["log_path"] = "./log_real_128/ex6n/ex7n/"
    ex7n_valid["log_file"] = "plx64"
    ex7n_valid["train"] = False
    ex7n_valid["eval"] = True
    ex7n_valid["test"] = False
    ex7n_valid["proof"] = False
    ex7n_valid["restore_file"] = "plx64-300.meta"  # keep empty to not restore a model.
    ex7n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex7n_valid["plot_title"] = "plx128_overfit: ex7n"
    ex7n_valid["label"] = "Experiment.py"
    if name == "ex7n_valid":
        name = ex7n_valid

    ex7n = {}
    ex7n["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex7n["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex7n["solver"] = Solver_nrm_cm
    ex7n["model"] = Model_8n
    ex7n["learning_rate"] = 0.001
    ex7n["num_iterations"] = 200
    ex7n["debug_output"] = True
    ex7n["log_path"] = "./log_real_128/ex6n/ex7n/"
    ex7n["log_file"] = "plx64"
    ex7n["train"] = True
    ex7n["eval"] = False
    ex7n["test"] = False
    ex7n["proof"] = False
    ex7n["stage2"] = True
    ex7n["only_r"] = False
    ex7n["cont"] = False
    ex7n["drp_cnv"] = 0.8
    ex7n["drp_pose"] = 0.9
    ex7n["restore_file"] = "plx64-200.meta"  # keep empty to not restore a model.
    ex7n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex7n["plot_title"] = "plx128_overfit: ex7n"
    ex7n["label"] = "Experiment.py"
    if name == "ex7n":
        name = ex7n

    ex8n_valid = {}
    ex8n_valid["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex8n_valid["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex8n_valid["solver"] = solver_nvalid
    ex8n_valid["model"] = Model_8n
    ex8n_valid["learning_rate"] = 0.001
    ex8n_valid["num_iterations"] = 100
    ex8n_valid["debug_output"] = True
    ex8n_valid["log_path"] = "./log_real_128/ex8n/"
    ex8n_valid["log_file"] = "plx64"
    ex8n_valid["train"] = False
    ex8n_valid["eval"] = True
    ex8n_valid["test"] = False
    ex8n_valid["proof"] = False
    ex8n_valid["restore_file"] = "plx64-200.meta"  # keep empty to not restore a model.
    ex8n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex8n_valid["plot_title"] = "plx128_overfit: ex8n"
    ex8n_valid["label"] = "Experiment.py"
    if name == "ex8n_valid":
        name = ex8n_valid

    ex8n = {}
    ex8n["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex8n["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex8n["solver"] = Solver_nrm_cm
    ex8n["model"] = Model_8n
    ex8n["learning_rate"] = 0.003
    ex8n["num_iterations"] = 100
    ex8n["debug_output"] = True
    ex8n["log_path"] = "./log_real_128/ex8n/"
    ex8n["log_file"] = "plx64"
    ex8n["train"] = True
    ex8n["eval"] = False
    ex8n["norm"] = True
    ex8n["test"] = False
    ex8n["proof"] = False
    ex8n["stage2"] = False
    ex8n["only_r"] = False
    ex8n["cont"] = False
    ex8n["drp_cnv"] = 0.90
    ex8n["drp_pose"] = 0.90
    ex8n["restore_file"] = "plx64-100.meta"  # keep empty to not restore a model.
    ex8n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex8n["plot_title"] = "plx128_overfit: ex8n"
    ex8n["label"] = "Experiment.py"
    if name == "ex8n":
        name = ex8n

    ex9n_valid = {}
    ex9n_valid["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex9n_valid["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex9n_valid["solver"] = solver_nvalid
    ex9n_valid["model"] = Model_8n
    ex9n_valid["learning_rate"] = 0.001
    ex9n_valid["num_iterations"] = 100
    ex9n_valid["debug_output"] = True
    ex9n_valid["log_path"] = "./log_real_128/ex8n/ex9n/"
    ex9n_valid["log_file"] = "plx64"
    ex9n_valid["train"] = False
    ex9n_valid["eval"] = True
    ex9n_valid["test"] = False
    ex9n_valid["proof"] = False
    ex9n_valid["restore_file"] = "plx64-600.meta"  # keep empty to not restore a model.
    ex9n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex9n_valid["plot_title"] = "plx128_overfit: ex9n"
    ex9n_valid["label"] = "Experiment.py"
    if name == "ex9n_valid":
        name = ex9n_valid

    ex9n = {}
    ex9n["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex9n["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex9n["solver"] = Solver_nrm_cm
    ex9n["model"] = Model_8n
    ex9n["learning_rate"] = 0.001
    ex9n["num_iterations"] = 200
    ex9n["debug_output"] = True
    ex9n["log_path"] = "./log_real_128/ex8n/ex9n/"
    ex9n["log_file"] = "plx64"
    ex9n["train"] = True
    ex9n["eval"] = False
    ex9n["test"] = False
    ex9n["proof"] = False
    ex9n["stage2"] = True
    ex9n["only_r"] = False
    ex9n["cont"] = False
    ex9n["drp_cnv"] = 0.95
    ex9n["drp_pose"] = 0.95
    ex9n["restore_file"] = "plx64-400.meta"  # keep empty to not restore a model.
    ex9n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex9n["plot_title"] = "plx128_overfit: ex9n"
    ex9n["label"] = "Experiment.py"
    if name == "ex9n":
        name = ex9n

    ex10n_valid = {}
    ex10n_valid["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex10n_valid["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex10n_valid["solver"] = solver_nvalid
    ex10n_valid["model"] = Model_8n
    ex10n_valid["learning_rate"] = 0.001
    ex10n_valid["num_iterations"] = 100
    ex10n_valid["debug_output"] = True
    ex10n_valid["log_path"] = "./log_real_128/ex10n/"
    ex10n_valid["log_file"] = "plx64"
    ex10n_valid["train"] = False
    ex10n_valid["eval"] = True
    ex10n_valid["test"] = False
    ex10n_valid["proof"] = False
    ex10n_valid["restore_file"] = "plx64-200.meta"  # keep empty to not restore a model.
    ex10n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex10n_valid["plot_title"] = "plx128_overfit: ex10n"
    ex10n_valid["label"] = "Experiment.py"
    if name == "ex10n_valid":
        name = ex10n_valid

    ex10n = {}
    ex10n["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex10n["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex10n["solver"] = Solver_nrm_cm
    ex10n["model"] = Model_8n
    ex10n["learning_rate"] = 0.003
    ex10n["num_iterations"] = 100
    ex10n["debug_output"] = True
    ex10n["log_path"] = "./log_real_128/ex10n/"
    ex10n["log_file"] = "plx64"
    ex10n["train"] = True
    ex10n["eval"] = False
    ex10n["test"] = False
    ex10n["proof"] = False
    ex10n["stage2"] = False
    ex10n["only_r"] = False
    ex10n["cont"] = False
    ex10n["drp_cnv"] = 0.85
    ex10n["drp_pose"] = 0.9
    ex10n["restore_file"] = "plx64-100.meta"  # keep empty to not restore a model.
    ex10n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex10n["plot_title"] = "plx128_overfit: ex10n"
    ex10n["label"] = "Experiment.py"
    if name == "ex10n":
        name = ex10n

    ex11n_valid = {}
    ex11n_valid["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex11n_valid["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex11n_valid["solver"] = solver_nvalid
    ex11n_valid["model"] = Model_8n
    ex11n_valid["learning_rate"] = 0.001
    ex11n_valid["num_iterations"] = 100
    ex11n_valid["debug_output"] = True
    ex11n_valid["log_path"] = "./log_real_128/ex10n/ex11n/"
    ex11n_valid["log_file"] = "plx64"
    ex11n_valid["train"] = False
    ex11n_valid["eval"] = True
    ex11n_valid["test"] = False
    ex11n_valid["proof"] = False
    ex11n_valid["restore_file"] = "plx64-400.meta"  # keep empty to not restore a model.
    ex11n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex11n_valid["plot_title"] = "plx128_overfit: ex11n"
    ex11n_valid["label"] = "Experiment.py"
    if name == "ex11n_valid":
        name = ex11n_valid

    ex11n = {}
    ex11n["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex11n["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex11n["solver"] = Solver_nrm_cm
    ex11n["model"] = Model_8n
    ex11n["learning_rate"] = 0.001
    ex11n["num_iterations"] = 200
    ex11n["debug_output"] = True
    ex11n["log_path"] = "./log_real_128/ex10n/ex11n/"
    ex11n["log_file"] = "plx64"
    ex11n["train"] = True
    ex11n["eval"] = False
    ex11n["test"] = False
    ex11n["proof"] = False
    ex11n["stage2"] = True
    ex11n["only_r"] = False
    ex11n["cont"] = False
    ex11n["drp_cnv"] = 0.8
    ex11n["drp_pose"] = 0.9
    ex11n["restore_file"] = "plx64-200.meta"  # keep empty to not restore a model.
    ex11n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex11n["plot_title"] = "plx128_overfit: ex11n"
    ex11n["label"] = "Experiment.py"
    if name == "ex11n":
        name = ex11n

    ex12n_valid = {}
    ex12n_valid["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex12n_valid["eval_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex12n_valid["solver"] = solver_nvalid
    ex12n_valid["model"] = Model_8n
    ex12n_valid["learning_rate"] = 0.001
    ex12n_valid["num_iterations"] = 100
    ex12n_valid["debug_output"] = True
    ex12n_valid["log_path"] = "./log_real_128/ex12n/"
    ex12n_valid["log_file"] = "plx64"
    ex12n_valid["train"] = False
    ex12n_valid["eval"] = True
    ex12n_valid["test"] = False
    ex12n_valid["proof"] = False
    ex12n_valid["restore_file"] = "plx64-200.meta"  # keep empty to not restore a model.
    ex12n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex12n_valid["plot_title"] = "plx128_overfit: ex12n"
    ex12n_valid["label"] = "Experiment.py"
    if name == "ex12n_valid":
        name = ex12n_valid

    ex12n = {}
    ex12n["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex12n["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex12n["solver"] = Solver_nrm_cm
    ex12n["model"] = Model_8n
    ex12n["learning_rate"] = 0.001
    ex12n["num_iterations"] = 100
    ex12n["debug_output"] = True
    ex12n["log_path"] = "./log_real_128/ex12n/"
    ex12n["log_file"] = "plx64"
    ex12n["train"] = True
    ex12n["eval"] = False
    ex12n["test"] = False
    ex12n["proof"] = False
    ex12n["stage2"] = False
    ex12n["only_r"] = False
    ex12n["cont"] = False
    ex12n["drp_cnv"] = 0.8
    ex12n["drp_pose"] = 0.9
    ex12n["restore_file"] = "plx64-100.meta"  # keep empty to not restore a model.
    ex12n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex12n["plot_title"] = "plx128_overfit: ex12n"
    ex12n["label"] = "Experiment.py"
    if name == "ex12n":
        name = ex12n

    ex13n_valid = {}
    ex13n_valid["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex13n_valid["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex13n_valid["solver"] = solver_nvalid
    ex13n_valid["model"] = Model_8n
    ex13n_valid["learning_rate"] = 0.001
    ex13n_valid["num_iterations"] = 100
    ex13n_valid["debug_output"] = True
    ex13n_valid["log_path"] = "./log_real_128/ex13n/"
    ex13n_valid["log_file"] = "plx64"
    ex13n_valid["train"] = False
    ex13n_valid["eval"] = True
    ex13n_valid["test"] = False
    ex13n_valid["proof"] = False
    ex13n_valid["restore_file"] = "plx64-500.meta"  # keep empty to not restore a model.
    ex13n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex13n_valid["plot_title"] = "plx128_overfit: ex13n"
    ex13n_valid["label"] = "Experiment.py"
    if name == "ex13n_valid":
        name = ex13n_valid

    ex13n = {}
    ex13n["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex13n["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex13n["solver"] = Solver_nrm
    ex13n["model"] = Model_8n
    ex13n["learning_rate"] = 0.001
    ex13n["num_iterations"] = 300
    ex13n["debug_output"] = True
    ex13n["log_path"] = "./log_real_128/ex13n/"
    ex13n["log_file"] = "plx64"
    ex13n["train"] = True
    ex13n["eval"] = False
    ex13n["test"] = False
    ex13n["proof"] = False
    ex13n["stage2"] = False
    ex13n["only_r"] = False
    ex13n["cont"] = True
    ex13n["drp_cnv"] = 0.85
    ex13n["drp_pose"] = 0.9
    ex13n["restore_file"] = "plx64-200.meta"  # keep empty to not restore a model.
    ex13n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex13n["plot_title"] = "plx128_overfit: ex13n"
    ex13n["label"] = "Experiment.py"
    if name == "ex13n":
        name = ex13n

    ex14n_valid = {}
    ex14n_valid["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex14n_valid["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex14n_valid["solver"] = solver_nvalid
    ex14n_valid["model"] = Model_8n
    ex14n_valid["learning_rate"] = 0.001
    ex14n_valid["num_iterations"] = 100
    ex14n_valid["debug_output"] = True
    ex14n_valid["log_path"] = "./log_real_128/ex14n/"
    ex14n_valid["log_file"] = "plx64"
    ex14n_valid["train"] = False
    ex14n_valid["eval"] = True
    ex14n_valid["test"] = False
    ex14n_valid["proof"] = False
    ex14n_valid["restore_file"] = "plx64-100.meta"  # keep empty to not restore a model.
    ex14n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex14n_valid["plot_title"] = "plx128_overfit: ex14n"
    ex14n_valid["label"] = "Experiment.py"
    if name == "ex14n_valid":
        name = ex14n_valid

    ex14n = {}
    ex14n["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex14n["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex14n["solver"] = Solver_nrm_cm
    ex14n["model"] = Model_8n
    ex14n["learning_rate"] = 0.001
    ex14n["num_iterations"] = 100
    ex14n["debug_output"] = True
    ex14n["log_path"] = "./log_real_128/ex14n/"
    ex14n["log_file"] = "plx64"
    ex14n["train"] = True
    ex14n["eval"] = False
    ex14n["test"] = False
    ex14n["proof"] = False
    ex14n["stage2"] = False
    ex14n["only_r"] = False
    ex14n["cont"] = False
    ex14n["drp_cnv"] = 0.85
    ex14n["drp_pose"] = 0.9
    ex14n["restore_file"] = ""  # keep empty to not restore a model.
    ex14n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex14n["plot_title"] = "plx128_overfit: ex14n"
    ex14n["label"] = "Experiment.py"
    if name == "ex14n":
        name = ex14n

    ex15n_valid = {}
    ex15n_valid["train_dataset"] = "../../datasets/real-5k-128.pickle"
    ex15n_valid["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex15n_valid["solver"] = solver_nvalid
    ex15n_valid["model"] = Model_8n
    ex15n_valid["learning_rate"] = 0.001
    ex15n_valid["num_iterations"] = 100
    ex15n_valid["debug_output"] = True
    ex15n_valid["log_path"] = "./log_real_128/ex15n/"
    ex15n_valid["log_file"] = "plx64"
    ex15n_valid["train"] = False
    ex15n_valid["eval"] = True
    ex15n_valid["test"] = False
    ex15n_valid["proof"] = False
    ex15n_valid["restore_file"] = "plx64-100.meta"  # keep empty to not restore a model.
    ex15n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex15n_valid["plot_title"] = "plx128_overfit: ex15n"
    ex15n_valid["label"] = "Experiment.py"
    if name == "ex15n_valid":
        name = ex15n_valid

    ex15n = {}
    ex15n["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex15n["eval_dataset"] = "../../datasets/real-5k-128.pickle"
    ex15n["solver"] = Solver_nrm_cm
    ex15n["model"] = Model_8n
    ex15n["learning_rate"] = 0.001
    ex15n["num_iterations"] = 100
    ex15n["debug_output"] = True
    ex15n["log_path"] = "./log_real_128/ex15n/"
    ex15n["log_file"] = "plx64"
    ex15n["train"] = True
    ex15n["eval"] = False
    ex15n["test"] = False
    ex15n["proof"] = False
    ex15n["stage2"] = False
    ex15n["only_r"] = False
    ex15n["cont"] = False
    ex15n["drp_cnv"] = 0.85
    ex15n["drp_pose"] = 0.9
    ex15n["restore_file"] = ""  # keep empty to not restore a model.
    ex15n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex15n["plot_title"] = "plx128_overfit: ex15n"
    ex15n["label"] = "Experiment.py"
    if name == "ex15n":
        name = ex15n
    return name


