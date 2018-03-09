import matplotlib.pyplot as plt
import numpy as np

phi = np.loadtxt('phi').flatten('F').tolist()
psi = np.loadtxt('psi').reshape(7, 20, 28).flatten('F').tolist()

for L in [phi, psi]:
    for i, item in enumerate(L):
        print '{}, '.format(item),
        if i % 5 == 4:
            print '&'
    print '\n'

plt.plot(range(28), phi.T)
#plt.show()
