#!/usr/bin/env python
"""

https://github.com/numpy/numpy/pull/19388

"""

import numpy as np

a = np.array(
      [b'TO AB                                                                                           ',
       b'TO BT AB                                                                                        ',
       b'TO BT BR AB                                                                                     ',
       b'TO BT BR BT AB                                                                                  ',
       b'TO BT BR BT BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BR BT BT BT ',
       b'TO SC SC SC SC SC SC SC SC SC BT BT BT BT BT BT BR BT BT BT BT BT BT SC BT BT DR BT BT BT BT SC ',
       b'TO SC SC SC SC SC SC SC SC SC RE BT BT BT BT BT BT BR BT BT BT BT BT BT SC BT BT BT BT BT BT BR ',
       b'TO SC SC SC SC SC SC SC SC SC RE BT BT BT BT SD                                                 ',
       b'TO SC SC SC SC SC SC SC SC SC SC AB                                                             ',
       b'TO SC SC SC SC SC SC SC SC SC SC BT AB                                                          '], dtype='|S96')

print(a)

idx = (a == a[6]).argmax()  ## because a == a[6] is boolean array can (in pricipal) shortcircuit at first True 
assert( idx == 6 )

import opticks.ana.idxstring as idxstring

