import numpy as np
import tensorflow as tf
import keras
from time import time as timer
import keras_tuner

from time import time as timer

from inputFile import inputFile
from meanfield import simulateLikelihood as smlt

from spoke_model import countMatrixModel as cmm 

inputSet = inputFile.CInputSet("gavin2006.csv", cmm.CountMatrixModel)

Nk = 300
nProteins = inputSet.observationG.nProteins
cmfa = smlt.CMeanFieldAnnealing(nProteins, Nk) # default

def likelihood_loss(x):
    mle = cmfa.estimate(inputSet.observationG, nProteins, Nk, x)  
    (fn, fp, total_loss) = cmfa.computeErrorRate(x, inputSet.observationG, nProteins)
    regularization_penalty = (1.0 + x)      
    regularized_loss = total_loss + regularization_penalty # this loss needs to be minimized
    print(regularized_loss)
    return regularized_loss

class RandomSearchTuner(keras_tuner.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        # Get the hp from trial.
        hp = trial.hyperparameters
        # Define "x" as a hyperparameter.
        x = hp.Float("psi", min_value=1.0, max_value=10.0)
        # Return the objective value to minimize.
        return likelihood_loss(x)

class BayesianTuner(keras_tuner.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        # Get the hp from trial.
        hp = trial.hyperparameters
        # Define "x" as a hyperparameter.
        x = hp.Float("psi", min_value=1.0, max_value=5.0)
        # Return the objective value to minimize.
        return likelihood_loss(x)

tuner = BayesianTuner(
    # No hypermodel or objective specified.
    max_trials=10,
    overwrite=True,
    directory="my_dir",
    project_name="random_search_tune",
)

# No need to pass anything to search()
# unless you use them in run_trial().
tuner.search(
    # Use the TensorBoard callback.
    # The logs will be write to "/tmp/tb_logs".
    callbacks=[keras.callbacks.TensorBoard("/tmp/tb_logs")])
print(tuner.get_best_hyperparameters()[0])
