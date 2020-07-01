#!/usr/bin/env python
"""
OpticksGenstepEnum.py
=======================

"""

import os
from collections import OrderedDict as odict
from opticks.ana.base import ini_


class OpticksGenstepEnum(object):
    def __init__(self):
        ini = ini_("$OPTICKS_PREFIX/include/OpticksCore/OpticksGenstep_Enum.ini")

        code2name = odict() 
        pfx = "OpticksGenstep_"
        for kv in sorted(ini.items(),key=lambda kv:int(kv[1])):
            assert kv[0].startswith(pfx)
            code2name[int(kv[1])] = kv[0][len(pfx):] 
        pass
        self.code2name = code2name

    def __call__(self, icode):
        return self.code2name.get(icode, "INVALID CODE")

    def __repr__(self):
        return "\n".join(["%2d : %s " % (int(kv[0]), kv[1]) for kv in self.code2name.items()])


if __name__ == '__main__':
    oge = OpticksGenstepEnum()
    print(oge)

