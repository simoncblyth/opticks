
#!/usr/bin/env python
"""

"""
import os, logging, numpy as np
log = logging.getLogger(__name__)


from env.numerics.npy.ana import Evt



if __name__ == '__main__':



    typ = "torch" 
    tag = "4" 
    det = "dayabay" 
    cat = "PmtInBox"  

    evt = Evt(tag=tag, src=typ, det=cat)
    print evt




