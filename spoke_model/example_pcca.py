import cmdtools as cmd
import cmdtools.systems.diffusion as dif
import matplotlib.pyplot as plt
import numpy as np

m=dif.TripleWell(nx=8, ny=8, beta=10)
plt.imshow(m.u)
plt.show()
Q=m.Q
p = cmd.analysis.pcca.PCCA(Q,3)
def plotchi(chi):
    plt.figure()
    plt.imshow(np.reshape(chi, np.shape(m.u)))
plotchi(p.chi[:,0])
plotchi(p.chi[:,1])
plotchi(p.chi[:,2])

plt.show()

