# from solvers.SolverRGBD_QuatLoss import *
import sys
sys.dont_write_bytecode = True
from models.M1 import *
from models.M2 import *
from models.m_sub.SM1 import *
from solvers.Solver_reim_cm import *


# This experiment works with the bunny model 128 x 128 pixels.
# The model color is adapted (chromatic adaptation) and noise was applied with sigma = 0.15
def exs(name):
    ex_test1 = {}
    ex_test1["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_test1["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_test1["solver"] = Solver_reim_cm
    ex_test1["model"] = M1
    ex_test1["batch"] = True
    ex_test1["learning_rate"] = 0.001
    ex_test1["num_iterations"] = 100
    ex_test1["log_path"] = "./log_batch/ex_test1/"
    ex_test1["log_file"] = "plx128"
    ex_test1["train"] = True
    ex_test1["proof"] = True
    ex_test1["stage2"] = False
    ex_test1["drp_cnv"] = 0.0
    if name == "ex_test1":
        name = ex_test1

    ex_test2 = {}
    ex_test2["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_test2["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_test2["solver"] = Solver_reim_cm
    ex_test2["model"] = M2
    ex_test2["batch"] = True
    ex_test2["learning_rate"] = 0.001
    ex_test2["num_iterations"] = 100
    ex_test2["log_path"] = "./log_batch/ex_test2/"
    ex_test2["log_file"] = "plx128"
    ex_test2["train"] = True
    ex_test2["proof"] = True
    ex_test2["stage2"] = False
    ex_test2["drp_cnv"] = 0.0
    if name == "ex_test2":
        name = ex_test2

    ex_test3 = {}
    ex_test3["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_test3["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_test3["solver"] = Solver_reim_cm
    ex_test3["model"] = SM1
    ex_test3["batch"] = True
    ex_test3["learning_rate"] = 0.001
    ex_test3["num_iterations"] = 100
    ex_test3["log_path"] = "./log_batch/ex_test3/"
    ex_test3["log_file"] = "plx128"
    ex_test3["train"] = True
    ex_test3["proof"] = True
    ex_test3["stage2"] = False
    ex_test3["drp_cnv"] = 0.1
    if name == "ex_test3":
        name = ex_test3

    ex_test4 = {}
    ex_test4["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_test4["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_test4["solver"] = Solver_reim_cm
    ex_test4["model"] = M2
    ex_test4["batch"] = True
    ex_test4["learning_rate"] = 0.001
    ex_test4["num_iterations"] = 100
    ex_test4["log_path"] = "./log_batch/ex_test4/"
    ex_test4["log_file"] = "plx128"
    ex_test4["train"] = True
    ex_test4["proof"] = True
    ex_test4["stage2"] = False
    ex_test4["drp_cnv"] = 0.1
    if name == "ex_test4":
        name = ex_test4


    return name


