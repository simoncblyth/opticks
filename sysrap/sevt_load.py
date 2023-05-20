#!/usr/bin/env python

import os, numpy as np
log = logging.getLogger(__name__)
from opticks.sysrap.sevt import SEvt

NEVT = int(os.environ.get("NEVT", 0))  # when NEVT>0 SEvt.LoadConcat loads and concatenates the SEvt


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    log.info("FOLD:%s" % os.environ.get("FOLD", "-"))
    t = SEvt.Load(symbol="t", NEVT=NEVT)
    print(repr(t))



