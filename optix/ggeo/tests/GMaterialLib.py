#!/usr/bin/env python
import os
import numpy as np

path_ = lambda _:os.path.expandvars("$IDPATH/%s" % _)
load_ = lambda _:np.load(path_(_))


def test_buffers():
    path = path_("GMaterialLib/GMaterialLib.npy")
    assert os.path.exists(path)
    os.system("ls -l %s " % path)
    buf = np.load(path)
    print "%100s %s " % (path, repr(buf.shape))
    print buf 
    return buf


if __name__ == '__main__':
    b = test_buffers()


