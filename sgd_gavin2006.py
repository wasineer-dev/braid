import numpy as np
from time import time as timer

import tensorflow as tf
from inputFile import inputFile
from meanfield import simulateLikelihood as smlt

from spoke_model import countMatrixModel as cmm 

inputSet = inputFile.CInputSet("gavin2002.csv", cmm.CountMatrixModel)

tf.config.run_functions_eagerly(True)

class MrfModel(tf.keras.Model):
    def __init__(self, nn_block = None, **kwargs):
        kwargs.setdefault("name", "custom_model")
        super().__init__(**kwargs)
        self.psi = tf.Variable(3.0)
        self.Nk = tf.constant(300)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        # Automatically resets the metric states at the start of each epoch or at the start of evaluate()
        return [self.loss_tracker]
        
    def compute_loss(self):
        Nk = self.Nk.numpy()
        nProteins = inputSet.observationG.nProteins
        cmfa = smlt.CMeanFieldAnnealing(nProteins, Nk) # default
        mle = cmfa.estimate(inputSet.observationG, nProteins, Nk, self.psi)  
        indicatorVec = tf.math.argmax(cmfa.tQ, axis=1)
        cmfa.mIndicatorQ = cmfa.tQ.numpy()
        cmfa.find_argmax()
        (fn, fp, fvalue) = cmfa.computeErrorRate(self.psi, cmfa.indicatorVec, inputSet.observationG, nProteins)
        #inputSet.writeCluster2File("my_dir/out_%.2f.tsv" % x, cmfa.mIndicatorQ, cmfa.indicatorVec)
        #inputSet.observationG.write2cytoscape("my_dir/out_%.2f.sif" % x, cmfa.indicatorVec, cmfa.mIndicatorQ, inputSet.aSortedProteins)    
        regularized_loss = fvalue  # this loss needs to be minimized
        print("%.8f" % regularized_loss)
        return tf.Variable(regularized_loss)

    def likelihood_loss(self):
        
        @tf.custom_gradient
        def loss_fn():
            y = self.compute_loss()
            def loss_gradient(upstream_grad, variables):
                dyByDx = upstream_grad
                return None, dyByDx    
            return y, loss_gradient

        loss = loss_fn()
        return loss
    
    def train_step(self, inputSet):
        if not self.trainable_weights:
            _ = self()  # required to initialize layer parameters with build() method
        with tf.GradientTape() as tape:
            loss = self.likelihood_loss()
        grads = tape.gradient(loss, self.trainable_weights)
        # Tell the optimizer to apply gradients on specified variables
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Update the running loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def call(self, x):
        return self.compute_loss(inputSet)

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

model = MrfModel(name="nn_logistic_model")
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4))
model.fit(batched_dataset, epochs=1, callbacks=[ReportWeightsCallback()])

