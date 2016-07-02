#!/usr/bin/env python
"""
For grabbing ggv arguments for checking 
"""
import numpy as np
from env.numerics.npy.prism import Prism, Box
import argparse

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--save", action="store_true", default=False ) 
    parser.add_argument("--test", action="store_true", default=False ) 
    parser.add_argument("--torch", action="store_true", default=False ) 
    parser.add_argument("--tag", default="" ) 
    parser.add_argument("--testconfig", default="" ) 
    parser.add_argument("--torchconfig", default="" ) 
    parser.add_argument("--animtimemax", default=100 ) 

    args = parser.parse_args()
    return args

kv_ = lambda s:map(lambda _:_.split("="),s.split("_"))


class Torch(object):
    def __init__(self, config):
        self.config = kv_(config)
        self.source = None
        self.target = None
        for k,v in self.config:
            if k == "source":
                self.source = np.fromstring(v, sep=",") 
            elif k == "target":
                self.target = np.fromstring(v, sep=",") 
            else:
                pass
            pass
        pass
        self.direction = self.target - self.source


    def __repr__(self):
        return "\n".join([
                      "source %25s " % self.source,
                      "target %25s " % self.target,
                      "direction %25s " % self.direction
                        ]) 

    def __str__(self):
        return "\n".join(["%20s : %s " % (k,v) for k,v in self.config])



class Test(object):
    def __init__(self, config):
        self.config = kv_(config)

        shapes = []
        boundaries = []
        parameters = []

        for k,v in self.config:
            if k == "shape":
                shapes.append(v)
            elif k == "boundary":
                boundaries.append(v)
            elif k == "parameters":
                parameters.append(v)
            else:
                pass

        assert len(shapes) == len(boundaries) == len(parameters)
        self.shapes = []
        for i in range(len(shapes)):
            shape = None
            if shapes[i] == "box":
                shape = Box(parameters[i], boundaries[i])
            elif shapes[i] == "prism":
                shape = Prism(parameters[i], boundaries[i])
            else:
                assert 0
            pass
            self.shapes.append(shape)

    def __str__(self):
        return "\n".join(map(str, self.shapes))
 
    def __repr__(self):
        return "\n".join(["%20s : %s " % (k,v) for k,v in self.config])






if __name__ == '__main__':
    #print "\n".join(sys.argv)
    args = parse_args()

    torch = Torch(args.torchconfig) 
    test = Test(args.testconfig) 

    sh = test.shapes[-1]

    print "torch:\n", torch
    print repr(torch)
    print "test:\n", test

    print "sh:\n", sh


    




