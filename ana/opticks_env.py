#!/usr/bin/env python

import os, logging, subprocess
log = logging.getLogger(__name__)

def subprocess_output(args):
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.communicate()

class OpticksEnv(object):
    """
    Hmm maybe an opticks exe to dump some JSON ? 
    Hmm but its default args ?
    """
    @classmethod
    def IDPATH(cls):
        exe = "OpticksIDPATH"  ## from okc-/tests
        out, err = subprocess_output([exe])
        out = out.strip()

        log.debug("[%s]" % out )
        if os.path.isdir(out):
            idpath = out 
            log.debug("found idpath dir %s " % idpath )
        else:
            idpath = None
            log.error("response from %s did not provide existing directory : out  %s err %s " % (cmd, out, err))
        pass      
        return idpath ; 

    @classmethod
    def OPTICKS_EVENT_BASE(cls):
        return os.path.expandvars("/tmp/$USER/opticks")

    @classmethod
    def Set(cls):
        os.environ.setdefault("IDPATH",cls.IDPATH())
        os.environ.setdefault("OPTICKS_EVENT_BASE",cls.OPTICKS_EVENT_BASE())

        
       
if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)
    OpticksEnv.Set()

    print "%s" % os.path.expandvars("$IDPATH/helo") 






