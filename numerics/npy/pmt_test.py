#!/usr/bin/env python
"""
PmtInBox Opticks vs cfg4
==================================

Without and with cfg4 runs::

   ggv-;ggv-pmt-test 
   ggv-;ggv-pmt-test --cfg4 

Visualize the cfg4 created evt in interop mode viewer::

   ggv-;ggv-pmt-test --cfg4 --load

Issues
-------

* source issue in cfg4 (b), allmost all photons doing same thing, 
  fixed by handling discLinear

* different G4 geometry, photon interaction positions in the interop visualization 
  show that primitives are there, but Z offsets not positioned correctly so the 
  boolean processing produces a more complicated geometry 

  * modify cfg4-/Detector to make just the Pyrex for simplification 

* after first order fixing G4 geometry to look OK, 
  still very different sequence histories because are missing surface/SD
  handling that leads to great simplification for Opticks as most photons 
  are absorbed/detected on the photocathode

* suspect the DYB detdesc G4 positioning of the photocathode inside the vacuum 
  with coincident surfaces will lead to comparison problems, as this "feature"
  was fixed for the surface-based translation  

  May need to compare against a correspondingly "fixed" G4 geometry



Very different::

    In [54]: a.history_table()
    Evt(1,"torch","PmtInBox","", seqs="[]")
                              noname 
                     8cd       375203       [3 ] TO BT SA
                     7cd       118603       [3 ] TO BT SD
                     8bd         4458       [3 ] TO BR SA
                     4cd         1471       [3 ] TO BT AB
                      4d           87       [2 ] TO AB
                 8cccccd           78       [7 ] TO BT BT BT BT BT SA
                     86d           68       [3 ] TO SC SA
                    8c6d           15       [4 ] TO SC BT SA
                  8bcccd            5       [6 ] TO BT BT BT BR SA
              ccccccbccd            3       [10] TO BT BT BR BT BT BT BT BT BT
                    4ccd            2       [4 ] TO BT BT AB
                    86bd            2       [4 ] TO BR SC SA
               8cccbbccd            1       [9 ] TO BT BT BR BR BT BT BT SA
                   8c6cd            1       [5 ] TO BT SC BT SA
                  8c6ccd            1       [6 ] TO BT BT SC BT SA
                     4bd            1       [3 ] TO BR AB
                    7c6d            1       [4 ] TO SC BT SD
                              500000 


    In [3]: b.history_table()
    Evt(-1,"torch","PmtInBox","", seqs="[]")
                              noname 
                8ccccccd       276675       [8 ] TO BT BT BT BT BT BT SA
                 8ccbccd       157768       [7 ] TO BT BT BR BT BT SA
              cccccccccd        14167       [10] TO BT BT BT BT BT BT BT BT BT
              ccccbccccd        13398       [10] TO BT BT BT BT BR BT BT BT BT
                     8bd         5397       [3 ] TO BR SA
              cbcccccccd         5153       [10] TO BT BT BT BT BT BT BT BR BT
              bbbbcccccd         4528       [10] TO BT BT BT BT BT BR BR BR BR
                  8ccbcd         4033       [6 ] TO BT BR BT BT SA
              cccbcccccd         3316       [10] TO BT BT BT BT BT BR BT BT BT
               8cccccccd         2700       [9 ] TO BT BT BT BT BT BT BT SA
              ccbcbcbccd         1895       [10] TO BT BT BR BT BR BT BR BT BT
                     4cd         1741       [3 ] TO BT AB




"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from env.numerics.npy.evt import Evt

X,Y,Z,W = 0,1,2,3


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(precision=4, linewidth=200)

    plt.ion()
    plt.close()

    tag = "1"
    a = Evt(tag="%s" % tag, src="torch", det="PmtInBox")
    b = Evt(tag="-%s" % tag , src="torch", det="PmtInBox")




