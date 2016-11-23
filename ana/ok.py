#!/usr/bin/env python

import os, sys, logging, numpy as np
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main


if __name__ == '__main__':
    ok = opticks_main()

    print ok.utag
    print ok.qwns




