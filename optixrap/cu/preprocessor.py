#!/usr/bin/env python
"""
preprocessor.py
================

A preprocessor that acts upon only a selction 
of flag macros, eg -ANGULAR_ENABLED,+WAY_ENABLED 

"""

import os, re, sys, logging, argparse
log = logging.getLogger(__name__)


class Preprocessor(object):
    ptn = re.compile("^#ifdef (?P<tag>\S*)")
    els = "#else"
    end = "#endif"
    fmt = "[%3d][%-50s][%-120s]" 

    def __init__(self, path, flags=None, dump=False, lines=[]):
        if dump:
            print(self.fmt % (-1,",".join(flags),""))
        pass
        rlines = open(path,"r").read().splitlines()
        if len(lines) == 0:
            lines = list(range(len(rlines)))
        elif len(lines) == 2:
            lines = list(range(lines[0],lines[1]))
        else:
            pass
        pass
        self.tags = []
        self.plines = []
        for i in lines:
            line = rlines[i]
            tag = self.parse_line(line)
            self.add_tag(tag)

            pline = self.process_line(line, self.tags, flags)
            if dump:
                print(self.fmt % (i, self.tagdesc, pline[0:120]))
            pass
            self.plines.append(pline)

            # #TAG %TAG @TAG only live for one line before decaying into +TAG -TAG or being removed  

            ltag = self.ltag
            if not ltag is None: 
                if ltag[0] == "#":                   # #TAG -> +TAG  
                    idx = self.tags.index(ltag)
                    self.tags[idx] = "+"+ltag[1:]
                elif ltag[0] == "%":                 # %TAG -> -TAG  
                    idx = self.tags.index(ltag)
                    self.tags[idx] = "-"+ltag[1:]
                elif ltag[0] == "@":                 # @TAG -> removed
                    self.tags.remove(ltag)
                else:
                    pass
                pass 
            pass
        pass
    pass

    def parse_line(self, line):
        """
        1. #ifdef TAG -> [+TAG] 
        2. #else      -> [%TAG]     
        3. #endif     -> [@TAG]
        4. otherwise last tag is returned
        """ 
        m = self.ptn.match(line)
        ltag = self.ltag
        with_ltag = not ltag is None 
        if m:
            tag = "#" + m.groupdict()["tag"]
        elif line.startswith(self.els):
            assert with_ltag and ltag[0] in "#+", line
            tag = "%" + ltag[1:] 
        elif line.startswith(self.end):
            assert with_ltag, line
            tag = "@" + ltag[1:]
        else:
            tag = ltag
        pass
        return tag


    def add_tag(self, tag):
        """
        1. -TAG evicts corresponding +TAG
        2. %TAG evicts corresponding +TAG
        2. @TAG evicts corresponding +TAG or -TAG 
        3. tags are added only when not already present
        """
        if tag is None: return
        assert tag[0] in "#+%-@" 
        ptag = "+"+tag[1:]
        mtag = "-"+tag[1:] 

        if tag[0] == "-":
            if ptag in self.tags:
                self.tags.remove(ptag)
            pass
        elif tag[0] == "%":
            if ptag in self.tags:
                self.tags.remove(ptag)
            pass
        elif tag[0] == "@":
            if ptag in self.tags:
                self.tags.remove(ptag)
            pass
            if mtag in self.tags:
                self.tags.remove(mtag)
            pass
        pass
        if not tag in self.tags:
            self.tags.append(tag) 
        pass


    def process_line(self, line, tags, flags): 
        """
        :param line: original 
        :param tags: tags assigned to the line, eg #TAG +TAG -TAG @TAG
        :param flags: input choice of macro flags to control preprocessor output 

        1. Default is to return the line asis
        2. if the stripped tags for the line 

        """
        sflags = list(set(map(lambda flag:flag[1:], flags)))  # strip the flags 
        for tag in tags:
            stag = tag[1:]
            if stag in sflags:  ## stripped tag matches one of the stripped flags 
                if tag[0] in "#%@":
                    line = "//// %s " % line  
                elif tag[0] in "+-":
                    if tag in flags:
                        line = "%s " % line  
                    else: 
                        line = "//// %s " % line  
                    pass   
                else:
                    assert 0 
                pass   
            pass
        pass
        return line 

    def _get_tagdesc(self):
        return ",".join(self.tags) 
    tagdesc = property(_get_tagdesc)

    def _get_ltag(self):
        return self.tags[-1] if len(self.tags) > 0 else None
    pass
    ltag = property(_get_ltag)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument( "path", help="File path to preprocess" )
    parser.add_argument( "-d", "--dump",    action="store_true", help="Dump the input file showing the tags assigned to each line and process result." )
    parser.add_argument( "-f", "--flags", help="Comma delimited control flags eg +WAY_ENABLED,-ANGULAR_ENABLED " )
    parser.add_argument( "-l", "--lines", help="Comma delimited line range", default=None )
    parser.add_argument( "-o", "--out",   help="Path to write the output", default=None )
    args = parser.parse_args()
    log.info("path:%s" % args.path)
    log.info("flags:%s" % args.flags)

    if args.lines is None:
        args.lines = []
    else:
        args.lines = list(map(lambda _:int(_), args.lines.split(",")))
    pass

    pp = Preprocessor(args.path, flags=args.flags.split(","), dump=args.dump, lines=args.lines ) 
    if not args.out:
        print("\n".join(pp.plines))
    else:
        log.info("writing to %s " % args.out )
        open(args.out,"w").write("\n".join(pp.plines)) 
    pass



