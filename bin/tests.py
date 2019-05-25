#!/usr/bin/env python

import os, sys, re

if __name__ == '__main__':

    ptn = re.compile("^    (\S*Test).cc\s*$")
    for line in file(sys.argv[1]).read().split("\n"):
        m = ptn.match(line)
        if m is not None:
            print(m.groups(1)[0])
