#!/usr/bin/env python

import logging 
from opticks.ana.base import opticks_environment
from opticks.ana.evt import Evt

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    opticks_environment()

    spol, ppol = "5", "6"

    p_devt = dict(tag=ppol, det="rainbow", label="P")
    #s_devt = dict(tag=spol, det="rainbow", label="S")

    p_evt = Evt(**p_devt)
    #s_evt = Evt(**s_devt)

    p_evt.history_table()
    #s_evt.history_table()



