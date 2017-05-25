#!/usr/bin/env python

import numpy as np

try: 
    from hashlib import md5 
except ImportError: 
    from md5 import md5 


def test_hello():
    s = 'hello'
    dig = md5()
    dig.update(s)
    print s, dig.hexdigest()


def test_array_digest():
    i = np.eye(4, dtype=np.float32)
    a = np.vstack([i,i,i]).reshape(-1,4,4)
    print array_digest(a)

if __name__ == '__main__':
    test_hello()
    test_array_digest()





    
