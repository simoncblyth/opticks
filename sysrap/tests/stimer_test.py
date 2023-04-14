#!/usr/bin/env python

import os, numpy as np

if __name__ == '__main__':
    tt = np.load(os.environ["TTPATH"])
    print(tt)

    expr = "np.c_[tt.view('datetime64[us]')]"
    print("\n%s\n"% expr )
    print(eval(expr))

