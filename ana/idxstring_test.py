#!/usr/bin/env python

import opticks.ana.idxstring as idxstring
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

af = np.asfortranarray(a)
print("a",a)

for i in range(len(a)):
    v = a[i] 
    idx = idxstring.find_first(v, af)
    assert( idx == i)
pass



