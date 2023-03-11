#!/usr/bin/env python

import numpy as np
import builtins

class Base(object):
    def __setattr__(self, name:str, value):    
        self.__dict__[name] = value
        pubpfx = getattr(self, "pubpfx", "DUMMY")   
        if name.startswith(pubpfx):
            print("publish pubpfx %s attribute to builtins %s:%s" % (pubpfx, name, value))
            setattr(builtins, name, value)
        pass
    pass

class Demo(Base):
    def __init__(self, pubpfx="mt"):
        self.pubpfx = pubpfx
        hello = "world"
        mt_hello = "world"

        self.hello = hello
        self.mt_hello = mt_hello
    pass


if __name__ == '__main__':
    d = Demo(pubpfx="mt")



