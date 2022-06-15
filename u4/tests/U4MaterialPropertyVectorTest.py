#!/usr/bin/env python

import os
import numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':

    Air = Fold.Load("$FOLD/Air", symbol="Air")
    Water = Fold.Load("$FOLD/Water", symbol="Water")
    Rock = Fold.Load("$FOLD/Rock", symbol="Rock")

    print(Air)
    print(Water)
    print(Rock)


