#!/usr/bin/env python
"""
OpticksFlags.py
=================

Used from optickscore/CMakeLists.txt

"""
import os, re, logging, argparse, json

## json_save_ duplicates opticks.ana.base 
## to make this script self contained 
## as this is used from the okc- build 

def makedirs_(path):
    pdir = os.path.dirname(path)
    if not os.path.exists(pdir):
        os.makedirs(pdir)
    pass
    return path 

expand_ = lambda path:os.path.expandvars(os.path.expanduser(path))
json_load_ = lambda path:json.load(file(expand_(path)))
json_save_ = lambda path, d:json.dump(d, file(makedirs_(expand_(path)),"w"))



log = logging.getLogger(__name__) 

class OpticksFlags(object):
    """
    """  
    pfx = "const char* OpticksFlags::_" 
    ptn = re.compile("^(?P<key>\S*)\s*=\s*\"(?P<val>\S{2})\"\s*;\s*$")

    @classmethod
    def Flag2Abbrev(cls, lines):
        """ 
        Apply pattern to lines starting with the prefix
        and match to extract the flag name and abbreviation.
 
            BULK_SCATTER      = "SC" ; 

        """  
        d = dict()  
        for line in lines:
            if not line.startswith(cls.pfx): continue
            line = line[len(cls.pfx):]
            m = cls.ptn.match(line)
            if not m: 
               log.debug("failed to match %s " % line )
               continue   
            pass
            g = m.groupdict()
            k, v = g["key"], g["val"]
            d[k] = v  
        pass
        return d

    def __repr__(self):
        return "\n".join([" %2s : %s " % (kv[1], kv[0]) for kv in self.flag2abbrev.items()])
 
    def __init__(self, path):
        path = os.path.expandvars(path)
        lines = file(path).readlines() 
        self.flag2abbrev = self.Flag2Abbrev(lines)

if __name__ == '__main__':

    default_path = "$OPTICKS_HOME/optickscore/OpticksFlags.cc"
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(     "path",  nargs="?", help="Path to input OpticksFlags.cc", default=default_path )
    parser.add_argument(     "--jsonpath", default=None, help="When a path is provided an json file will be written to it." ) 
    parser.add_argument(     "--quiet", action="store_true", default=False, help="Skip dumping" ) 
    parser.add_argument(     "--level", default="info", help="logging level" ) 
    args = parser.parse_args()

    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)

    if args.path == default_path:
        log.info("using default input path %s " % args.path)
    else:
        log.info("using argument input path %s " % args.path)
    pass  

    flags = OpticksFlags(args.path)

    if not args.quiet:  
        print(flags)
    pass 
    if not args.jsonpath is None:
        log.info("writing flag2abbrev to jsonpath %s " % args.jsonpath)
        json_save_(args.jsonpath, flags.flag2abbrev )  
    pass



