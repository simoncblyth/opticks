#!/usr/bin/env python
"""
Visualize the cfg4 created evt in interop mode viewer::

   ggv-;ggv-pmt-test --cfg4 --load

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

    In [55]: b.his
    b.history        b.history_table  

    In [55]: b.history_table()
    Evt(-1,"torch","PmtInBox","", seqs="[]")
                              noname 
                  8ccccd       480108       [6 ] TO BT BT BT BT SA
              ccccccbccd        16934       [10] TO BT BT BR BT BT BT BT BT BT
                   4cccd         1207       [5 ] TO BT BT BT AB
              cccccbcccd          823       [10] TO BT BT BT BR BT BT BT BT BT
              cbccccbccd          577       [10] TO BT BT BR BT BT BT BT BR BT
                  4ccccd           89       [6 ] TO BT BT BT BT AB
                 86ccccd           83       [7 ] TO BT BT BT BT SC SA
              bcccccbccd           31       [10] TO BT BT BR BT BT BT BT BT BR
              4cccccbccd           27       [10] TO BT BT BR BT BT BT BT BT AB
                 8ccc6cd           23       [7 ] TO BT SC BT BT BT SA
                8cbbcccd           23       [8 ] TO BT BT BT BR BR BT SA
                 8cccc6d           15       [7 ] TO SC BT BT BT BT SA
                      4d           12       [2 ] TO AB
               8ccccc6cd           11       [9 ] TO BT SC BT BT BT BT BT SA
              8cbc6ccccd           10       [10] TO BT BT BT BT SC BT BR BT SA
              cccc6ccccd           10       [10] TO BT BT BT BT SC BT BT BT BT
                     4cd            6       [3 ] TO BT AB
              8ccccbc6cd            2       [10] TO BT SC BT BR BT BT BT BT SA
              ccc6ccbccd            2       [10] TO BT BT BR BT BT SC BT BT BT
                    4ccd            2       [4 ] TO BT BT AB
                8cb6cccd            1       [8 ] TO BT BT BT SC BR BT SA
              c6cccbcccd            1       [10] TO BT BT BT BR BT BT BT SC BT
              cccc6cbccd            1       [10] TO BT BT BR BT SC BT BT BT BT
                  4bcccd            1       [6 ] TO BT BT BT BR AB
              ccccc6cccd            1       [10] TO BT BT BT SC BT BT BT BT BT
                              500000 


Looks like source issue in cfg4 (b), allmost all photons doing same thing::

    In [56]: a.ox
    Out[56]: 
    A([[[  68.3786,   27.8457,  104.5609,    0.9754],
            [  -0.0501,   -0.0204,   -0.9985,    1.    ],
            [  -0.3772,    0.9262,    0.    ,  380.    ],
            [      nan,    0.    ,    0.    ,    0.    ]],

    In [57]: b.ox
    Out[57]: 
    A([[[   0.    ,    0.    , -300.    ,    1.3503],
            [   0.    ,    0.    ,   -1.    ,    1.    ],
            [   1.    ,    0.    ,    0.    ,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],


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




