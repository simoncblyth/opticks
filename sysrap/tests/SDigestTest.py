#!/usr/bin/env python

import os, numpy as np

try: 
    from hashlib import md5 
except ImportError: 
    from md5 import md5 


from opticks.ana.nbase import array_digest 



def test_hello():
    s = 'hello'
    dig = md5()
    dig.update(s)
    print s, dig.hexdigest()


def test_array_digest():
    """
    digest on the file includes the header, but array_digest 
    covers just the data
    """
    i = np.eye(4, dtype=np.float32)
    a = np.vstack([i,i,i]).reshape(-1,4,4)
    print array_digest(a)
    np.save(os.path.expandvars("$TMP/test_array_digest.npy"), a )



if __name__ == '__main__':
    test_hello()
    test_array_digest()





    
