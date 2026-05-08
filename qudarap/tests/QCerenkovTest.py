#!/usr/bin/env python
"""
QCerenkovTest.py
================

::

    ~/o/qudarap/tests/QCerenkovTest.sh
    ~/o/qudarap/tests/QCerenkovTest.sh pdb


"""
import os, logging, numpy as np
log = logging.getLogger(__name__)
from opticks.ana.fold import Fold


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    f = Fold.Load(symbol="f")
    print(repr(f))

    # THIS ICDF APPROACH NO LONGER USED FOR CERENKOV
    #match = np.all( f.cerenkov_icdf == f.cerenkov_lookup )
    #assert match



