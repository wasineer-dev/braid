import numpy as np
from time import time as timer

import tensorflow as tf
from inputFile import inputFile
from meanfield import simulateLikelihood as smlt

from spoke_model import countMatrixModel as cmm 

inputSet = inputFile.CInputSet("gavin2002.csv", cmm.CountMatrixModel)
Nk = tf.constant(300)
tf.config.run_functions_eagerly(True)

@tf.custom_gradient
def loss_fn(psi):
    nProteins = inputSet.observationG.nProteins
    cmfa = smlt.CMeanFieldAnnealing(nProteins, Nk) # default
    mle = cmfa.estimate(inputSet.observationG, nProteins, Nk, psi)  
    indicatorVec = tf.math.argmax(cmfa.tQ, axis=1)
    cmfa.mIndicatorQ = cmfa.tQ.numpy()
    cmfa.find_argmax()
    (fn, fp, fvalue) = cmfa.computeErrorRate(psi, cmfa.indicatorVec, inputSet.observationG, nProteins)
    #inputSet.writeCluster2File("my_dir/out_%.2f.tsv" % x, cmfa.mIndicatorQ, cmfa.indicatorVec)
    #inputSet.observationG.write2cytoscape("my_dir/out_%.2f.sif" % x, cmfa.indicatorVec, cmfa.mIndicatorQ, inputSet.aSortedProteins)    
    regularized_loss = np.float32(fvalue)  # this loss needs to be minimized
    print("%.8f" % regularized_loss)
    def loss_gradient(upstream_grad):
        gradient = 2.0
        return upstream_grad * gradient    
    return regularized_loss, loss_gradient

class Mrf_Layer(tf.keras.layers.Layer):
    def __init__(self):
        super(Mrf_Layer, self).__init__()

    def call(self, psi):
        return loss_fn(psi)
        
mrf_layer = Mrf_Layer()

class MrfModel(tf.keras.Model):
    def __init__(self, nn_block = None, **kwargs):
        kwargs.setdefault("name", "custom_model")
        super().__init__(**kwargs)
        self.psi = tf.Variable(3., dtype=tf.float32)
        self.Nk = tf.constant(300)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        
    @property
    def metrics(self):
        # Automatically resets the metric states at the start of each epoch or at the start of evaluate()
        return [self.loss_tracker]
    
    def train_step(self, inputSet):
        if not self.trainable_weights:
            _ = self()  # required to initialize layer parameters with build() method

        # Open a GradientTape.
        with tf.GradientTape() as tape:
            current_loss = mrf_layer(self.psi)

        grads = tape.gradient(current_loss, self.trainable_weights)
        for g in grads:
            print(g)
        # Tell the optimizer to apply gradients on specified variables
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Update the running loss
        self.loss_tracker.update_state(current_loss)
        return {"loss": self.loss_tracker.result()}

    def call(self, x):
        return self.mrf_layer(self.psi)

class ReportWeightsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        print(self.model.get_weights())

inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)
for batch in batched_dataset.take(4):
  print([arr.numpy() for arr in batch])

model = MrfModel(name="my_model")
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4))
model.fit(batched_dataset, epochs=1, callbacks=[ReportWeightsCallback()])

