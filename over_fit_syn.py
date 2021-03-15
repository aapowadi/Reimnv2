#from solvers.SolverRGBD_QuatLoss import *
import sys
sys.dont_write_bytecode = True
from models.Model_8n import *
from solvers.Solver2StageRGBD_6DoF import *
from solvers.Solver2StageRGBD_6DoF_b import *
from solvers.Solver_nrm import *
from solvers.solver_valid import *
from solvers.solver_nvalid import *

# This experiment works with the bunny model 128 x 128 pixels.
# The model color is adapted (chromatic adaptation) and noise was applied with sigma = 0.15
def exs(name):

    ex1n_valid = {}
    ex1n_valid["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex1n_valid["test_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex1n_valid["solver"] = solver_nvalid
    ex1n_valid["model"] = Model_8n
    ex1n_valid["learning_rate"] = 0.1
    ex1n_valid["num_iterations"] = 100
    ex1n_valid["debug_output"] = True
    ex1n_valid["log_path"] = "./log_syn_overfit/ex1n/"
    ex1n_valid["log_file"] = "plx64"
    ex1n_valid["train"] = False
    ex1n_valid["eval"] = True
    ex1n_valid["test"] = False
    ex1n_valid["proof"] = False
    ex1n_valid["restore_file"] = "plx64-100.meta"  # keep empty to not restore a model.
    ex1n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex1n_valid["plot_title"] = "plx64_overfit: ex1n"
    ex1n_valid["label"] = "Experiment.py"
    if name == "ex1n_valid":
        name = ex1n_valid

    ex1n = {}
    ex1n["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex1n["test_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex1n["solver"] = Solver_nrm
    ex1n["model"] = Model_8n
    ex1n["learning_rate"] = 0.1
    ex1n["num_iterations"] = 100
    ex1n["debug_output"] = True
    ex1n["log_path"] = "./log_syn_overfit/ex1n/"
    ex1n["log_file"] = "plx64"
    ex1n["train"] = True
    ex1n["eval"] = False
    ex1n["test"] = False
    ex1n["proof"] = False
    ex1n["stage2"] = False
    ex1n["cont"] = False
    ex1n["drp_cnv"] = 1.0
    ex1n["drp_pose"] = 1.0
    ex1n["restore_file"] = ""  # keep empty to not restore a model.
    ex1n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex1n["plot_title"] = "plx64_overfit: ex1n"
    ex1n["label"] = "Experiment.py"
    if name == "ex1n":
        name = ex1n

    ex2n_valid = {}
    ex2n_valid["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex2n_valid["test_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex2n_valid["solver"] = solver_nvalid
    ex2n_valid["model"] = Model_8n
    ex2n_valid["learning_rate"] = 0.01
    ex2n_valid["num_iterations"] = 100
    ex2n_valid["debug_output"] = True
    ex2n_valid["log_path"] = "./log_syn_overfit/ex2n/"
    ex2n_valid["log_file"] = "plx64"
    ex2n_valid["train"] = False
    ex2n_valid["eval"] = True
    ex2n_valid["test"] = False
    ex2n_valid["proof"] = False
    ex2n_valid["restore_file"] = "plx64-100.meta"  # keep empty to not restore a model.
    ex2n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex2n_valid["plot_title"] = "plx64_overfit: ex2n"
    ex2n_valid["label"] = "Experiment.py"
    if name == "ex2n_valid":
        name = ex2n_valid

    ex2n = {}
    ex2n["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex2n["test_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex2n["solver"] = Solver_nrm
    ex2n["model"] = Model_8n
    ex2n["learning_rate"] = 0.01
    ex2n["num_iterations"] = 100
    ex2n["debug_output"] = True
    ex2n["log_path"] = "./log_syn_overfit/ex2n/"
    ex2n["log_file"] = "plx64"
    ex2n["train"] = True
    ex2n["eval"] = False
    ex2n["test"] = False
    ex2n["proof"] = False
    ex2n["stage2"] = False
    ex2n["cont"] = False
    ex2n["drp_cnv"] = 1.0
    ex2n["drp_pose"] = 1.0
    ex2n["restore_file"] = ""  # keep empty to not restore a model.
    ex2n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex2n["plot_title"] = "plx64_overfit: ex2n"
    ex2n["label"] = "Experiment.py"
    if name == "ex2n":
        name = ex2n

    ex3n_valid = {}
    ex3n_valid["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex3n_valid["test_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex3n_valid["solver"] = solver_nvalid
    ex3n_valid["model"] = Model_8n
    ex3n_valid["learning_rate"] = 0.001
    ex3n_valid["num_iterations"] = 100
    ex3n_valid["debug_output"] = True
    ex3n_valid["log_path"] = "./log_syn_overfit/ex3n/"
    ex3n_valid["log_file"] = "plx64"
    ex3n_valid["train"] = False
    ex3n_valid["eval"] = True
    ex3n_valid["test"] = False
    ex3n_valid["proof"] = False
    ex3n_valid["restore_file"] = "plx64-100.meta"  # keep empty to not restore a model.
    ex3n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex3n_valid["plot_title"] = "plx64_overfit: ex3n"
    ex3n_valid["label"] = "Experiment.py"
    if name == "ex3n_valid":
        name = ex3n_valid

    ex3n = {}
    ex3n["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex3n["test_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex3n["solver"] = Solver_nrm
    ex3n["model"] = Model_8n
    ex3n["learning_rate"] = 0.001
    ex3n["num_iterations"] = 100
    ex3n["debug_output"] = True
    ex3n["log_path"] = "./log_syn_overfit/ex3n/"
    ex3n["log_file"] = "plx64"
    ex3n["train"] = True
    ex3n["eval"] = False
    ex3n["test"] = False
    ex3n["proof"] = False
    ex3n["stage2"] = False
    ex3n["cont"] = False
    ex3n["drp_cnv"] = 1.0
    ex3n["drp_pose"] = 1.0
    ex3n["restore_file"] = ""  # keep empty to not restore a model.
    ex3n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex3n["plot_title"] = "plx64_overfit: ex3n"
    ex3n["label"] = "Experiment.py"
    if name == "ex3n":
        name = ex3n

    ex4n_valid = {}
    ex4n_valid["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex4n_valid["test_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex4n_valid["solver"] = solver_nvalid
    ex4n_valid["model"] = Model_8n
    ex4n_valid["learning_rate"] = 0.004
    ex4n_valid["num_iterations"] = 100
    ex4n_valid["debug_output"] = True
    ex4n_valid["log_path"] = "./log_syn_overfit/ex4n/"
    ex4n_valid["log_file"] = "plx64"
    ex4n_valid["train"] = False
    ex4n_valid["eval"] = True
    ex4n_valid["test"] = False
    ex4n_valid["proof"] = False
    ex4n_valid["restore_file"] = "plx64-100.meta"  # keep empty to not restore a model.
    ex4n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex4n_valid["plot_title"] = "plx64_overfit: ex4n"
    ex4n_valid["label"] = "Experiment.py"
    if name == "ex4n_valid":
        name = ex4n_valid

    ex4n = {}
    ex4n["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex4n["test_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex4n["solver"] = Solver_nrm
    ex4n["model"] = Model_8n
    ex4n["learning_rate"] = 0.004
    ex4n["num_iterations"] = 100
    ex4n["debug_output"] = True
    ex4n["log_path"] = "./log_syn_overfit/ex4n/"
    ex4n["log_file"] = "plx64"
    ex4n["train"] = True
    ex4n["eval"] = False
    ex4n["test"] = False
    ex4n["proof"] = False
    ex4n["stage2"] = False
    ex4n["cont"] = False
    ex4n["drp_cnv"] = 1.0
    ex4n["drp_pose"] = 1.0
    ex4n["restore_file"] = ""  # keep empty to not restore a model.
    ex4n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex4n["plot_title"] = "plx64_overfit: ex4n"
    ex4n["label"] = "Experiment.py"
    if name == "ex4n":
        name = ex4n

    ex5n_valid = {}
    ex5n_valid["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex5n_valid["test_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex5n_valid["solver"] = solver_nvalid
    ex5n_valid["model"] = Model_8n
    ex5n_valid["learning_rate"] = 0.005
    ex5n_valid["num_iterations"] = 100
    ex5n_valid["debug_output"] = True
    ex5n_valid["log_path"] = "./log_syn_overfit/ex5n/"
    ex5n_valid["log_file"] = "plx64"
    ex5n_valid["train"] = False
    ex5n_valid["eval"] = True
    ex5n_valid["test"] = False
    ex5n_valid["proof"] = False
    ex5n_valid["restore_file"] = "plx64-100.meta"  # keep empty to not restore a model.
    ex5n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex5n_valid["plot_title"] = "plx64_overfit: ex5n"
    ex5n_valid["label"] = "Experiment.py"
    if name == "ex5n_valid":
        name = ex5n_valid

    ex5n = {}
    ex5n["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex5n["test_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex5n["solver"] = Solver_nrm
    ex5n["model"] = Model_8n
    ex5n["learning_rate"] = 0.005
    ex5n["num_iterations"] = 100
    ex5n["debug_output"] = True
    ex5n["log_path"] = "./log_syn_overfit/ex5n/"
    ex5n["log_file"] = "plx64"
    ex5n["train"] = True
    ex5n["eval"] = False
    ex5n["test"] = False
    ex5n["proof"] = False
    ex5n["stage2"] = False
    ex5n["cont"] = False
    ex5n["drp_cnv"] = 1.0
    ex5n["drp_pose"] = 1.0
    ex5n["restore_file"] = ""  # keep empty to not restore a model.
    ex5n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex5n["plot_title"] = "plx64_overfit: ex5n"
    ex5n["label"] = "Experiment.py"
    if name == "ex5n":
        name = ex5n

    ex6n_valid = {}
    ex6n_valid["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex6n_valid["test_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex6n_valid["solver"] = solver_nvalid
    ex6n_valid["model"] = Model_8n
    ex6n_valid["learning_rate"] = 0.006
    ex6n_valid["num_iterations"] = 100
    ex6n_valid["debug_output"] = True
    ex6n_valid["log_path"] = "./log_syn_overfit/ex6n/"
    ex6n_valid["log_file"] = "plx64"
    ex6n_valid["train"] = False
    ex6n_valid["eval"] = True
    ex6n_valid["test"] = False
    ex6n_valid["proof"] = False
    ex6n_valid["restore_file"] = "plx64-100.meta"  # keep empty to not restore a model.
    ex6n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex6n_valid["plot_title"] = "plx64_overfit: ex6n"
    ex6n_valid["label"] = "Experiment.py"
    if name == "ex6n_valid":
        name = ex6n_valid

    ex6n = {}
    ex6n["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex6n["test_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex6n["solver"] = Solver_nrm
    ex6n["model"] = Model_8n
    ex6n["learning_rate"] = 0.006
    ex6n["num_iterations"] = 100
    ex6n["debug_output"] = True
    ex6n["log_path"] = "./log_syn_overfit/ex6n/"
    ex6n["log_file"] = "plx64"
    ex6n["train"] = True
    ex6n["eval"] = False
    ex6n["test"] = False
    ex6n["proof"] = False
    ex6n["stage2"] = False
    ex6n["cont"] = False
    ex6n["drp_cnv"] = 1.0
    ex6n["drp_pose"] = 1.0
    ex6n["restore_file"] = ""  # keep empty to not restore a model.
    ex6n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex6n["plot_title"] = "plx64_overfit: ex6n"
    ex6n["label"] = "Experiment.py"
    if name == "ex6n":
        name = ex6n

    ex7n_valid = {}
    ex7n_valid["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex7n_valid["test_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex7n_valid["solver"] = solver_nvalid
    ex7n_valid["model"] = Model_8n
    ex7n_valid["learning_rate"] = 0.007
    ex7n_valid["num_iterations"] = 100
    ex7n_valid["debug_output"] = True
    ex7n_valid["log_path"] = "./log_syn_overfit/ex7n/"
    ex7n_valid["log_file"] = "plx64"
    ex7n_valid["train"] = False
    ex7n_valid["eval"] = True
    ex7n_valid["test"] = False
    ex7n_valid["proof"] = False
    ex7n_valid["restore_file"] = "plx64-100.meta"  # keep empty to not restore a model.
    ex7n_valid["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex7n_valid["plot_title"] = "plx64_overfit: ex7n"
    ex7n_valid["label"] = "Experiment.py"
    if name == "ex7n_valid":
        name = ex7n_valid

    ex7n = {}
    ex7n["train_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex7n["test_dataset"] = "../../datasets/tr-128-40k.pickle"
    ex7n["solver"] = Solver_nrm
    ex7n["model"] = Model_8n
    ex7n["learning_rate"] = 0.007
    ex7n["num_iterations"] = 100
    ex7n["debug_output"] = True
    ex7n["log_path"] = "./log_syn_overfit/ex7n/"
    ex7n["log_file"] = "plx64"
    ex7n["train"] = True
    ex7n["eval"] = False
    ex7n["test"] = False
    ex7n["proof"] = False
    ex7n["stage2"] = False
    ex7n["cont"] = False
    ex7n["drp_cnv"] = 1.0
    ex7n["drp_pose"] = 1.0
    ex7n["restore_file"] = ""  # keep empty to not restore a model.
    ex7n["quat_used"] = False  # set true, if the dataset contains quaternions. Otherwise false.
    ex7n["plot_title"] = "plx64_overfit: ex7n"
    ex7n["label"] = "Experiment.py"
    if name == "ex7n":
        name = ex7n

    return name


