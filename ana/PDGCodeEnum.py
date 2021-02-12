#!/usr/bin/env python
"""
PDGCodeEnum.py
=======================

::

   ~/opticks/ana/PDGCodeEnum.py

"""

from collections import OrderedDict as odict

class PDGCodeEnum(object):
    def __init__(self):
        code2name = odict()
        code2name[11] = "e-"
        code2name[13] = "mu-"
        code2name[-11] = "e+"
        code2name[-13] = "mu+"
        code2name[22] = "gamma"
        code2name[20022] = "photon"
        self.code2name = code2name

    def __call__(self, icode):
        return self.code2name.get(icode, "INVALID CODE")

    def __repr__(self):
        return "\n".join(["%2d : %s " % (int(kv[0]), kv[1]) for kv in self.code2name.items()])


if __name__ == '__main__':
    pce = PDGCodeEnum()
    print(pce)
    

