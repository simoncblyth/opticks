#!/usr/bin/env python

a = np.load("/tmp/animator.npy")
plt.plot(a[:,0,0], a[:,0,1])
