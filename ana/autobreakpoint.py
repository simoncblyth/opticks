#!/usr/bin/env python
"""
"""
from collections import OrderedDict
import os, re, fnmatch, logging

log = logging.getLogger(__name__)

def prep(TXT):
    return map(lambda line:line.lstrip().rstrip(), filter(None,TXT.split("\n")))


class BreakPoint(dict):
    TMPL = "\n".join(prep(r"""   
    # %(func)s 
    br set -f %(source)s -l %(line)s
    br com add %(index)s -F %(module)s.%(func)s
    """))
    INDEX = 0 

    @classmethod
    def resolve_path(cls, patn):
        base = os.environ["OPTICKS_HOME"]
        paths = []
        for path, dirs, files in os.walk(os.path.abspath(base)):
            for filename in fnmatch.filter(files, patn):
                paths.append(os.path.join(path, filename))
            pass
        pass
        if len(paths) != 1:
            #log.warning("failed to resove : %s " % patn )
            return None
        pass 
        return paths[0]

    @classmethod
    def resolve_bpline( cls, fnptn, mkr ):
        """
        Finds the source and looks for the marker, to give the lineno
        of the breakpoint.  Remember to recompile after changing source!
        """
        path = cls.resolve_path(fnptn)
        if path is None:
            return None
        pass
        marker = "// (*lldb*) %s" % mkr
        #print "marker:[%s]" % marker
        nls = filter( lambda nl:nl[1].find(marker) > -1, enumerate(file(path).readlines()) )
        l1 = int(nls[0][0]) + 1 if len(nls) == 1 else None
        return l1 
 

    def __init__(self, func, module):
        dict.__init__(self)
        elem = func.split("_")
        assert len(elem) == 4
        self.__class__.INDEX += 1

        self["func"] = func
        self["module"] = module
        self["index"] = self.__class__.INDEX
        self["source"] = "%s.%s" % (elem[0], elem[1])

        try:
            line = int(elem[2]) 
            self["line"] = line
            self["mkr"] = "NO"
        except ValueError:
            self["mkr"] = elem[2]
            self["line"] = "%(line)s"
        pass

        if type(self["line"]) is str:
            self["line"] = self.resolve_bpline(self["source"], self["mkr"])
        pass
        #print self

    def __str__(self):
        return self.TMPL % self



class AutoBreakPoint(dict):

    HEAD = r"""
    # AutoBreakPoint generated 
    # 
    # path       %(path)s 
    # module     %(module)s 
    # thisfile   %(thisfile)s 
    #

    command script import %(module)s
    """

    DEF_PTN = re.compile("^def (\S*)\(frame") 

    def parse(self, path, module):
        for line in file(path).readlines():
            m = self.DEF_PTN.search(line)
            if not m:continue
            func = m.group(1)
            elem = func.split("_")
            if len(elem) == 4:
                bp = BreakPoint(func, module)
                if bp["line"] is None:
                    pass
                    #log.warning("skipping bp func %s " % func )
                else:
                    self.bps.append(bp)
                pass
            pass
        pass

    def __init__(self, path, module):
        dict.__init__(self, path=path, module=module)
        self.bps = []
        self.parse(path, module)

    def __repr__(self):
        self["thisfile"] = os.path.abspath(__file__) 
        HEAD = self.HEAD % self
        pass
        return "\n".join(prep(HEAD)+map(str, self.bps))


if __name__ == '__main__':
    abp = AutoBreakPoint(path=__file__)
    print abp




