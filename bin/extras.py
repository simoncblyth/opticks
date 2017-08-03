#!/usr/bin/env python

from opticks.ana.base import opticks_main, dump_extras_meta
       
if __name__ == '__main__':
    args = opticks_main()
    base = "$OPTICKS_IDFOLD/extras"
    dump_extras_meta(base, name="meta.json", fmt=" %(idx)5s : %(height)6s : %(lvname)-40s : %(soname)-40s : %(err)s ")

   
    

