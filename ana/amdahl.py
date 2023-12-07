#!/usr/bin/env python

import os, numpy as np
import matplotlib.pyplot as plt
SIZE = np.array([1280, 720])


class Amdahl(object):
     """
     https://en.wikipedia.org/wiki/Amdahl%27s_law


                   1
            -------------------
             (1 - p)  +   p/F

     """
     @classmethod
     def Overall_Speedup(cls, F, p):
         """
         :param F: parallel speedup
         :param p: parallel fraction 
         """
         return 1./( (1-p) + p/F )


if __name__ == '__main__':

     fig, ax = plt.subplots(1, figsize=SIZE/100. )
     fig.suptitle("~/opticks/ana/amdahl.sh : Overall Speedup for Parallel Fractions, p ")

     F = np.linspace( 1, 10000 , 10000 )  
     PARALLEL_FRACTION = np.array([0.99, 0.985, 0.98, 0.97, 0.95, 0.90])

     COLORS = ["red","green","blue", "cyan", "magenta", "black", "yellow" ]

     for i, p in enumerate(PARALLEL_FRACTION): 
         color = COLORS[i%len(COLORS)]
         ax.plot( F, Amdahl.Overall_Speedup(F, p), label="p,1-p,1/(1-p): %5.3f %5.3f %5.3f " % (p, 1-p, 1/(1.-p) ), color=color  )
         ax.hlines( 1/(1.-p), xmin=F[0], xmax=F[-1], linestyle="dashed", color=color ) 
     pass

     ax.axvspan(100, 1000, alpha=0.1, color='blue')
     ax.set_xscale('log') 
     ax.set_ylabel("Overall Speedup (para frac. 0.90 -> 0.99)", fontsize=20 )
     ax.set_xlabel("Parallelized Speedup", fontsize=20 )
     ax.legend()

     fig.show()
    


