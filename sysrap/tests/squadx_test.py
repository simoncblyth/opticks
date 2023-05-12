#!/usr/bin/env python

import os, textwrap, numpy as np

if __name__ == '__main__':
    a = np.load(os.environ["SQUADX_TEST_PATH"])


    EXPRS = r"""
    
    a
   
    # comment

    a.view(np.uint64)

    a.view(np.uint64)  #hex

    a.view(np.uint64)

    a.view("datetime64[us]")

    """

    for e in textwrap.dedent(EXPRS).split("\n"):
        print(e)
        if len(e) == 0 or e[0] == "#": continue
        ifmt = hex if "#hex" in e else None 
        np.set_printoptions(formatter={'int':ifmt})
        print(eval(e))
    pass


