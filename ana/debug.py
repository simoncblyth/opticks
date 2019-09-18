#!/usr/bin/env python
"""
debug.py
==========

Unfortunately attempting::

   from opticks.ana.debug import MyPdb 

Gives "Name error cannot import", so have to duplicate the
below piece of code wherever wish to plant breakpoints for
consumption by Pdb.  

For a useful interactive stack dump presentation
at "set_trace" breakpoints or errors use ip:: 

    ip(){ 
        local py=${1:-dummy.py};
        ipython --pdb -i -- $(which $py) ${@:2}
    }

For example::

   ip tboolean.py --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show 

"""
import logging
log = logging.getLogger(__name__)

try:
    from IPython.core.debugger import Pdb as MyPdb
except ImportError:
    class MyPdb(object):
        def set_trace(self):
            log.error("IPython is required for ipdb.set_trace() " )
        pass  
    pass
pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ipdb = MyPdb()
    ipdb.set_trace()


