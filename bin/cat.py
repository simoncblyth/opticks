#!/usr/bin/env python

import sys, os

if __name__ == '__main__':
    path = sys.argv[1]
    assert os.path.exists(path)
    #print(len(sys.argv))
    slin = map(int,sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    lines = map(str.strip,open(path, "r").readlines())
    ilines = range(len(lines)) if slin is None else slin

    count_ = lambda line:len(filter(lambda _:_ == line, lines))

    print("\n".join(["%-4d :%d: %s" % (i, count_(lines[i]), lines[i]) for i in ilines]))




