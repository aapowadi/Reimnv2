from Experiment import *
from over_fit_syn import *
import sys, getopt

def main(argv):
   try:
      opts, args = getopt.getopt(argv,"hi:",["ifile="])
   except getopt.GetoptError:
      print('ex_syn.py -i <exp_name>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('ex_syn.py -i <exp_name>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
          ag = exs(arg)
          experiment = Experiment(ag)
          experiment.start()
if __name__ == "__main__":
   main(sys.argv[1:])