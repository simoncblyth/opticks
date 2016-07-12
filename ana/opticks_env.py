#!/usr/bin/env python


assert 0, "NOT IN USE"
import os, logging, subprocess
log = logging.getLogger(__name__)

def subprocess_output(args):
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.communicate()

class OpticksEnv(object):
    """
    Hmm maybe an opticks exe to dump some JSON ? 
    Hmm but its default args ?

    Cannot do this at OpticksIDPATH level 
    alone as geo selection relies on op.sh envvar setup.
    """
    @classmethod
    def IDPATH(cls):
        exe = "OpticksIDPATH"  ## from okc-/tests
        stdout, stderr = subprocess_output([exe])
        msg = stderr.strip()

        log.debug("[%s]" % msg )
        if os.path.isdir(msg):
            idpath = msg 
            log.debug("found idpath dir %s " % idpath )
        else:
            idpath = None
            log.error("response from %s did not provide existing directory : out  %s err %s " % (exe, stdout, stderr))
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






