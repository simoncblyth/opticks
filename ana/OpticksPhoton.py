#!/usr/bin/env python
"""
OpticksPhoton.py
=================

Formerly OpticksFlags.py 

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
json_load_ = lambda path:json.load(open(expand_(path), "r"))
json_save_ = lambda path, d:json.dump(d, open(makedirs_(expand_(path)),"w"))



log = logging.getLogger(__name__) 

class OpticksPhoton(object):
    pfx = "    static constexpr const char* _" 
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
            select = line.startswith(cls.pfx)
            log.debug( " select %d line %s " % (select, line ))
            if not select: continue
            line = line[len(cls.pfx):]
            m = cls.ptn.match(line)
            if not m: 
               log.debug("failed to match %s " % line )
               continue   
            pass
            g = m.groupdict()
            k, v = g["key"], g["val"]
            d[k] = v  
            log.debug( " k %20s v %s " % (k,v ))
        pass
        return d

    def __repr__(self):
        return "\n".join([" %2s : %s " % (kv[1], kv[0]) for kv in self.flag2abbrev.items()])
 
    def __init__(self, hh_path):
        hh_path = os.path.expandvars(hh_path)
        hh_lines = open(hh_path, "r").readlines() 
        log.debug(" hh_path %s hh_lines %d " % (hh_path, len(hh_lines)) )
        self.flag2abbrev = self.Flag2Abbrev(hh_lines)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(__doc__)
    default_path = "$OPTICKS_HOME/sysrap/OpticksPhoton.hh"
    parser.add_argument(     "path",  nargs="?", help="Path to input OpticksPhoton.hh", default=default_path )
    parser.add_argument(     "--jsonpath", default=None, help="When a path is provided an json file will be written to it." ) 
    parser.add_argument(     "--quiet", action="store_true", default=False, help="Skip dumping" ) 
    parser.add_argument(     "--level", default="info", help="logging level" ) 
    args = parser.parse_args()

    fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
    logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)

    flags = OpticksPhoton(args.path)

    if not args.quiet:  
        print(flags)
    pass 
    if not args.jsonpath is None:
        log.info("writing flag2abbrev to jsonpath %s " % args.jsonpath)
        json_save_(args.jsonpath, flags.flag2abbrev )  
    pass



