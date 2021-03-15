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
    ex_test_valid["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex_test_valid["eval_dataset"] = "../../datasets/tr-128-40k.pickle"
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
    ex_test_valid["restore_file"] = "plx64-110.meta"  # keep empty to not restore a model.
    ex_test_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex_test_valid["plot_title"] = "plx128_overfit: ex_test"
    ex_test_valid["label"] = "Experiment.py"
    if name == "ex_test_valid":
        name = ex_test_valid

    ex_test = {}
    ex_test["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex_test["eval_dataset"] = "../../datasets/tr-128-40k.pickle"
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
    ex_test["drp_cnv"] = 0.0
    ex_test["drp_pose"] = 0.0
    ex_test["restore_file"] = ""  # keep empty to not restore a model.
    ex_test["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex_test["plot_title"] = "plx128_overfit: ex_test"
    ex_test["label"] = "Experiment.py"
    if name == "ex_test":
        name = ex_test


    ex2n_valid = {}
    ex2n_valid["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex2n_valid["eval_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex2n_valid["solver"] = solver_nvalid
    ex2n_valid["model"] = Model_8n
    ex2n_valid["learning_rate"] = 0.001
    ex2n_valid["num_iterations"] = 100
    ex2n_valid["debug_output"] = True
    ex2n_valid["log_path"] = "./log_real_128/ex1n/ex2n/"
    ex2n_valid["log_file"] = "plx64"
    ex2n_valid["train"] = False
    ex2n_valid["eval"] = True
    ex2n_valid["test"] = False
    ex2n_valid["proof"] = False
    ex2n_valid["restore_file"] = "plx64-300.meta"  # keep empty to not restore a model.
    ex2n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex2n_valid["plot_title"] = "plx128_overfit: ex2n"
    ex2n_valid["label"] = "Experiment.py"
    if name == "ex2n_valid":
        name = ex2n_valid

    ex2n = {}
    ex2n["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex2n["eval_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex2n["solver"] = Solver_nrm_cm
    ex2n["model"] = Model_8n
    ex2n["learning_rate"] = 0.001
    ex2n["num_iterations"] = 100
    ex2n["debug_output"] = True
    ex2n["log_path"] = "./log_real_128/ex1n/ex2n/"
    ex2n["log_file"] = "plx64"
    ex2n["train"] = True
    ex2n["eval"] = False
    ex2n["test"] = False
    ex2n["proof"] = False
    ex2n["stage2"] = True
    ex2n["cont"] = False
    ex2n["drp_cnv"] = 0.9
    ex2n["drp_pose"] = 0.9
    ex2n["restore_file"] = "plx64-200.meta"  # keep empty to not restore a model.
    ex2n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex2n["plot_title"] = "plx128_overfit: ex2n"
    ex2n["label"] = "Experiment.py"
    if name == "ex2n":
        name = ex2n

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
    ex6n_valid["learning_rate"] = 0.003
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
    ex6n["solver"] = Solver_nrm
    ex6n["model"] = Model_8n
    ex6n["learning_rate"] = 0.003
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
    return name


