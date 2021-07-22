# from solvers.SolverRGBD_QuatLoss import *
import sys
sys.dont_write_bytecode = True
from models.m_sub.SM11 import *
from models.m_sub.SM22 import *
from models.m_sub.SM33 import *
from models.m_sub.SM44 import *
from solvers.Solver_reim_cm import *


# This experiment works with the bunny model 128 x 128 pixels.
# The model color is adapted (chromatic adaptation) and noise was applied with sigma = 0.15
def exs(name):

    ex_nn = {}
    ex_nn["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn["solver"] = Solver_reim_cm
    ex_nn["model"] = SM11
    ex_nn["learning_rate"] = 0.00001
    ex_nn["num_iterations"] = 100
    ex_nn["log_path"] = "./log_testnn/ex_nn/"
    ex_nn["trained_models"] = "./trained_models/testnn/ex_nn/"
    ex_nn["cont"] = True
    ex_nn["log_file"] = "plx128"
    ex_nn["train"] = True
    ex_nn["proof"] = False
    ex_nn["stage2"] = False
    ex_nn["drp_cnv"] = 0.2
    if name == "ex_nn":
        name = ex_nn

    ex_nn1 = {}
    ex_nn1["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn1["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn1["solver"] = Solver_reim_cm
    ex_nn1["model"] = SM22
    ex_nn1["learning_rate"] = 0.00001
    ex_nn1["num_iterations"] = 200
    ex_nn1["log_path"] = "./log_testnn/ex_nn1/"
    ex_nn1["trained_models"] = "./trained_models/testnn/ex_nn1/"
    ex_nn1["cont"] = False
    ex_nn1["log_file"] = "plx128_M8_batch_normalization"
    ex_nn1["train"] = True
    ex_nn1["proof"] = False
    ex_nn1["stage2"] = False
    ex_nn1["drp_cnv"] = 0.2
    if name == "ex_nn1":
        name = ex_nn1

    ex_nn2 = {}
    ex_nn2["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn2["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn2["solver"] = Solver_reim_cm
    ex_nn2["model"] = SM33
    ex_nn2["learning_rate"] = 0.00001
    ex_nn2["num_iterations"] = 100
    ex_nn2["log_path"] = "./log_testnn/ex_nn2/"
    ex_nn2["trained_models"] = "./trained_models/testnn/ex_nn2/"
    ex_nn2["log_file"] = "plx128_M8_upsampling_activations"
    ex_nn2["train"] = True
    ex_nn2["cont"] = False
    ex_nn2["proof"] = False
    ex_nn2["stage2"] = False
    ex_nn2["drp_cnv"] = 0.2
    if name == "ex_nn2":
        name = ex_nn2

    ex_nn3 = {}
    ex_nn3["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn3["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn3["solver"] = Solver_reim_cm
    ex_nn3["model"] = SM44
    ex_nn3["learning_rate"] = 0.00001
    ex_nn3["num_iterations"] = 100
    ex_nn3["log_path"] = "./log_testnn/ex_nn3/"
    ex_nn3["trained_models"] = "./trained_models/testnn/ex_nn3/"
    ex_nn3["log_file"] = "plx128_M8_upsampling_acts_batch_norm"
    ex_nn3["train"] = True
    ex_nn3["cont"] = False
    ex_nn3["proof"] = False
    ex_nn3["stage2"] = False
    ex_nn3["drp_cnv"] = 0.2
    if name == "ex_nn3":
        name = ex_nn3

    ex_nn4 = {}
    ex_nn4["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn4["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn4["solver"] = Solver_reim_cm
    ex_nn4["model"] = SM11
    ex_nn4["learning_rate"] = 0.0001
    ex_nn4["num_iterations"] = 100
    ex_nn4["log_path"] = "./log_testnn/ex_nn4/"
    ex_nn4["trained_models"] = "./trained_models/testnn/ex_nn4/"
    ex_nn4["log_file"] = "plx128_ex_nn_with_10p_more_dropout"
    ex_nn4["train"] = True
    ex_nn4["proof"] = False
    ex_nn4["stage2"] = False
    ex_nn4["drp_cnv"] = 0.3
    if name == "ex_nn4":
        name = ex_nn4

    ex_nn5 = {}
    ex_nn5["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn5["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn5["solver"] = Solver_reim_cm
    ex_nn5["model"] = SM11
    ex_nn5["learning_rate"] = 0.0001
    ex_nn5["num_iterations"] = 100
    ex_nn5["log_path"] = "./log_testnn/ex_nn5/"
    ex_nn5["trained_models"] = "./trained_models/testnn/ex_nn5/"
    ex_nn5["log_file"] = "plx128_ex_nn2_with_10p_more_dropout"
    ex_nn5["train"] = True
    ex_nn5["cont"] = False
    ex_nn5["proof"] = False
    ex_nn5["stage2"] = False
    ex_nn5["drp_cnv"] = 0.3
    if name == "ex_nn5":
        name = ex_nn5

    ex_nn6 = {}
    ex_nn6["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn6["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn6["solver"] = Solver_reim_cm
    ex_nn6["model"] = SM11
    ex_nn6["learning_rate"] = 0.0001
    ex_nn6["num_iterations"] = 100
    ex_nn6["log_path"] = "./log_testnn/ex_nn6/"
    ex_nn6["trained_models"] = "./trained_models/testnn/ex_nn6/"
    ex_nn6["log_file"] = "plx128_ex_nn_with_40p_dropout"
    ex_nn6["train"] = True
    ex_nn6["cont"] = False
    ex_nn6["proof"] = False
    ex_nn6["stage2"] = False
    ex_nn6["drp_cnv"] = 0.4
    if name == "ex_nn6":
        name = ex_nn6

    ex_nn7 = {}
    ex_nn7["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn7["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn7["solver"] = Solver_reim_cm
    ex_nn7["model"] = SM11
    ex_nn7["learning_rate"] = 0.0001
    ex_nn7["num_iterations"] = 100
    ex_nn7["log_path"] = "./log_testnn/ex_nn7/"
    ex_nn7["trained_models"] = "./trained_models/testnn/ex_nn7/"
    ex_nn7["log_file"] = "plx128_ex_nn2_with_40p_dropout"
    ex_nn7["train"] = True
    ex_nn7["cont"] = False
    ex_nn7["proof"] = False
    ex_nn7["stage2"] = False
    ex_nn7["drp_cnv"] = 0.4
    if name == "ex_nn7":
        name = ex_nn7

    ex_nn8 = {}
    ex_nn8["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn8["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn8["solver"] = Solver_reim_cm
    ex_nn8["model"] = SM22
    ex_nn8["learning_rate"] = 0.001
    ex_nn8["num_iterations"] = 100
    ex_nn8["log_path"] = "./log_testnn/ex_nn8/"
    ex_nn8["trained_models"] = "./trained_models/testnn/ex_nn8/"
    ex_nn8["log_file"] = "plx128"
    ex_nn8["train"] = True
    ex_nn8["cont"] = True
    ex_nn8["proof"] = False
    ex_nn8["stage2"] = False
    ex_nn8["drp_cnv"] = 0.2
    if name == "ex_nn8":
        name = ex_nn8

    ex_nn9 = {}
    ex_nn9["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn9["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn9["solver"] = Solver_reim_cm
    ex_nn9["model"] = SM22
    ex_nn9["learning_rate"] = 0.001
    ex_nn9["num_iterations"] = 100
    ex_nn9["log_path"] = "./log_testnn/ex_nn9/"
    ex_nn9["trained_models"] = "./trained_models/testnn/ex_nn9/"
    ex_nn9["log_file"] = "plx128"
    ex_nn9["train"] = True
    ex_nn9["cont"] = True
    ex_nn9["proof"] = False
    ex_nn9["stage2"] = False
    ex_nn9["drp_cnv"] = 0.3
    if name == "ex_nn9":
        name = ex_nn9

    ex_nn10 = {}
    ex_nn10["train_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn10["eval_dataset"] = "../../datasets/tr-128-comb.pickle"
    ex_nn10["solver"] = Solver_reim_cm
    ex_nn10["model"] = SM11
    ex_nn10["learning_rate"] = 0.001
    ex_nn10["num_iterations"] = 100
    ex_nn10["log_path"] = "./log_testnn/ex_nn10/"
    ex_nn10["trained_models"] = "./trained_models/testnn/ex_nn10/"
    ex_nn10["log_file"] = "plx128"
    ex_nn10["train"] = True
    ex_nn10["cont"] = False
    ex_nn10["proof"] = False
    ex_nn10["stage2"] = False
    ex_nn10["drp_cnv"] = 0.2
    if name == "ex_nn10":
        name = ex_nn10


    return name


