# from solvers.SolverRGBD_QuatLoss import *
import sys
sys.dont_write_bytecode = True
from models.m_sub.SM1 import *
from models.m_sub.SM2 import *
from models.m_sub.SM3 import *
from solvers.Solver_reim_cm import *


# This experiment works with the bunny model 128 x 128 pixels.
# The model color is adapted (chromatic adaptation) and noise was applied with sigma = 0.15
def exs(name):

    ex_n = {}
    ex_n["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n["solver"] = Solver_reim_cm
    ex_n["model"] = SM1
    ex_n["learning_rate"] = 0.0001
    ex_n["num_iterations"] = 100
    ex_n["log_path"] = "./log_test/ex_n/"
    ex_n["trained_models"] = "./trained_models/test/ex_n/"
    ex_n["cont"] = False
    ex_n["log_file"] = "plx128"
    ex_n["train"] = True
    ex_n["proof"] = False
    ex_n["stage2"] = False
    ex_n["drp_cnv"] = 0.2
    if name == "ex_n":
        name = ex_n

    ex_n1 = {}
    ex_n1["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n1["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n1["solver"] = Solver_reim_cm
    ex_n1["model"] = SM2
    ex_n1["learning_rate"] = 0.0001
    ex_n1["num_iterations"] = 100
    ex_n1["log_path"] = "./log_test/ex_n1/"
    ex_n1["trained_models"] = "./trained_models/test/ex_n1/"
    ex_n1["cont"] = True
    ex_n1["log_file"] = "plx128_M8_batch_normalization"
    ex_n1["train"] = True
    ex_n1["proof"] = False
    ex_n1["stage2"] = False
    ex_n1["drp_cnv"] = 0.2
    if name == "ex_n1":
        name = ex_n1

    ex_n2 = {}
    ex_n2["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n2["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n2["solver"] = Solver_reim_cm
    ex_n2["model"] = SM3
    ex_n2["learning_rate"] = 0.0001
    ex_n2["num_iterations"] = 100
    ex_n2["log_path"] = "./log_test/ex_n2/"
    ex_n2["trained_models"] = "./trained_models/test/ex_n2/"
    ex_n2["log_file"] = "plx128_M8_upsampling_activations"
    ex_n2["train"] = True
    ex_n2["cont"] = False
    ex_n2["proof"] = False
    ex_n2["stage2"] = False
    ex_n2["drp_cnv"] = 0.2
    if name == "ex_n2":
        name = ex_n2

    ex_n3 = {}
    ex_n3["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n3["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n3["solver"] = Solver_reim_cm
    ex_n3["model"] = SM1
    ex_n3["learning_rate"] = 0.001
    ex_n3["num_iterations"] = 100
    ex_n3["log_path"] = "./log_test/ex_n3/"
    ex_n3["trained_models"] = "./trained_models/test/ex_n3/"
    ex_n3["log_file"] = "plx128"
    ex_n3["train"] = True
    ex_n3["cont"] = True
    ex_n3["proof"] = False
    ex_n3["stage2"] = False
    ex_n3["drp_cnv"] = 0.3
    if name == "ex_n3":
        name = ex_n3

    ex_n4 = {}
    ex_n4["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n4["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n4["solver"] = Solver_reim_cm
    ex_n4["model"] = SM2
    ex_n4["learning_rate"] = 0.001
    ex_n4["num_iterations"] = 100
    ex_n4["log_path"] = "./log_test/ex_n4/"
    ex_n4["trained_models"] = "./trained_models/test/ex_n4/"
    ex_n4["log_file"] = "plx128"
    ex_n4["train"] = True
    ex_n4["proof"] = False
    ex_n4["stage2"] = False
    ex_n4["drp_cnv"] = 0.1
    if name == "ex_n4":
        name = ex_n4

    ex_n5 = {}
    ex_n5["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n5["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n5["solver"] = Solver_reim_cm
    ex_n5["model"] = SM2
    ex_n5["learning_rate"] = 0.001
    ex_n5["num_iterations"] = 100
    ex_n5["log_path"] = "./log_test/ex_n5/"
    ex_n5["trained_models"] = "./trained_models/test/ex_n5/"
    ex_n5["log_file"] = "plx128"
    ex_n5["train"] = True
    ex_n5["cont"] = True
    ex_n5["proof"] = False
    ex_n5["stage2"] = False
    ex_n5["drp_cnv"] = 0.2
    if name == "ex_n5":
        name = ex_n5

    ex_n6 = {}
    ex_n6["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n6["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n6["solver"] = Solver_reim_cm
    ex_n6["model"] = SM2
    ex_n6["learning_rate"] = 0.001
    ex_n6["num_iterations"] = 100
    ex_n6["log_path"] = "./log_test/ex_n6/"
    ex_n6["trained_models"] = "./trained_models/test/ex_n6/"
    ex_n6["log_file"] = "plx128"
    ex_n6["train"] = True
    ex_n6["cont"] = True
    ex_n6["proof"] = False
    ex_n6["stage2"] = False
    ex_n6["drp_cnv"] = 0.3
    if name == "ex_n6":
        name = ex_n6

    ex_n7 = {}
    ex_n7["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n7["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n7["solver"] = Solver_reim_cm
    ex_n7["model"] = SM2
    ex_n7["learning_rate"] = 0.001
    ex_n7["num_iterations"] = 100
    ex_n7["log_path"] = "./log_test/ex_n7/"
    ex_n7["trained_models"] = "./trained_models/test/ex_n7/"
    ex_n7["log_file"] = "plx128"
    ex_n7["train"] = True
    ex_n7["cont"] = True
    ex_n7["proof"] = False
    ex_n7["stage2"] = False
    ex_n7["drp_cnv"] = 0.1
    if name == "ex_n7":
        name = ex_n7

    ex_n8 = {}
    ex_n8["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n8["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n8["solver"] = Solver_reim_cm
    ex_n8["model"] = SM2
    ex_n8["learning_rate"] = 0.001
    ex_n8["num_iterations"] = 100
    ex_n8["log_path"] = "./log_test/ex_n8/"
    ex_n8["trained_models"] = "./trained_models/test/ex_n8/"
    ex_n8["log_file"] = "plx128"
    ex_n8["train"] = True
    ex_n8["cont"] = True
    ex_n8["proof"] = False
    ex_n8["stage2"] = False
    ex_n8["drp_cnv"] = 0.2
    if name == "ex_n8":
        name = ex_n8

    ex_n9 = {}
    ex_n9["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n9["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n9["solver"] = Solver_reim_cm
    ex_n9["model"] = SM2
    ex_n9["learning_rate"] = 0.001
    ex_n9["num_iterations"] = 100
    ex_n9["log_path"] = "./log_test/ex_n9/"
    ex_n9["trained_models"] = "./trained_models/test/ex_n9/"
    ex_n9["log_file"] = "plx128"
    ex_n9["train"] = True
    ex_n9["cont"] = True
    ex_n9["proof"] = False
    ex_n9["stage2"] = False
    ex_n9["drp_cnv"] = 0.3
    if name == "ex_n9":
        name = ex_n9

    ex_n10 = {}
    ex_n10["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n10["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_n10["solver"] = Solver_reim_cm
    ex_n10["model"] = SM3
    ex_n10["learning_rate"] = 0.001
    ex_n10["num_iterations"] = 100
    ex_n10["log_path"] = "./log_test/ex_n10/"
    ex_n10["trained_models"] = "./trained_models/test/ex_n10/"
    ex_n10["log_file"] = "plx128"
    ex_n10["train"] = True
    ex_n10["cont"] = False
    ex_n10["proof"] = False
    ex_n10["stage2"] = False
    ex_n10["drp_cnv"] = 0.2
    if name == "ex_n10":
        name = ex_n10


    return name


