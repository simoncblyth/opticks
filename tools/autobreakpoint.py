#!/usr/bin/env python
"""
AutoBreakPoint
================

Usage example from ana/g4lldb.py::

    256 if __name__ == '__main__':
    257     print AutoBreakPoint(path=__file__, module="opticks.ana.g4lldb")
    258 

Instanciating and printing to stdout the AutoBreakPoint repr provides the 
lldb commands to set breakpoints and add python function commands to them.
This works by parsing the source of the invoking file.
Only functions with names and arguments following the required
pattern yield breakpoints. Thus change function names 
to disable breakpoints, eg prefix with "_".

::

    def CRandomEngine_cc_postTrack(frame, bp_loc, sess):
        pass
    def G4SteppingManager_cc_191(frame, bp_loc, sess):
        pass

The name encodes breakpoint source name and line number or marker such as "postTrack". 
When markers are used the source is searched for a string such as "// (*lldb*) postTrack"
where marker is "postTrack" in order to resolve the line number. 
Markers have the advantage of remaining valid as the source is changed.

Start developing breakpoint function with something like::

    def CSteppingAction_cc_setStep(frame, bp_loc, sess):
        ploc = Loc(sys._getframe(), __name__)
        print "%s :  %s " % (ploc.tag, ploc.label)

        self = EV(frame.FindVariable("this"))
        print self
        stop = True 
        return stop   

Generation example::

    delta:ana blyth$ g4lldb.py 
    # AutoBreakPoint generated
    #
    # path       /Users/blyth/opticks/ana/g4lldb.py
    # module     opticks.ana.g4lldb
    # thisfile   /Users/blyth/opticks/tools/autobreakpoint.py
    #
    command script import opticks.ana.g4lldb

    # CRandomEngine_cc_preTrack
    br set -f CRandomEngine.cc -l 352
    br com add 1 -F opticks.ana.g4lldb.CRandomEngine_cc_preTrack
    ...


This generation is done automatically by bin/op.sh prior 
to Opticks launch when using the "-DD" option (see op-vi).

Example::

   tboolean-;tboolean-box --okg4 --align --mask 1230  --pindex 0 --pindexlog -DD  

      ## python scripted breakpoints are typically used 
      ## with masked running on single photons


Future Directions
--------------------

These breakpoint functions could easily be generated from 
the handling class, with source class being imported into the
generated python.

Hmm this is true, but for simple things you dont need a handling class.


::

    ENGINE = None
    def CRandomEngine_cc_preTrack(frame, bp_loc, sess):
        global ENGINE
        ENGINE = CRandomEngine()
        ploc = Loc(sys._getframe(), __name__)
        return ENGINE.preTrack(ploc, frame, bp_loc, sess)

    def CRandomEngine_cc_flat(frame, bp_loc, sess):
        ploc = Loc(sys._getframe(), __name__)
        return ENGINE.flat(ploc,frame, bp_loc, sess)
     


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
        """
        Example breakpoint func names: CRec_cc_add G4SteppingManager_cc_215
        """
        dict.__init__(self)
        elem = func.split("_")
        assert len(elem) == 3
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

    DEF_PTN = re.compile("^def (\S*)\(frame, bp_loc, sess\):$")

    def parse(self, path, module):
        for line in file(path).readlines():
            m = self.DEF_PTN.search(line)
            if not m:continue
            func = m.group(1)
            elem = func.split("_")
            if len(elem) == 3:
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
        return "\n".join(prep(HEAD)+map(str, self.bps))


if __name__ == '__main__':
    abp = AutoBreakPoint(path=__file__)
    print abp


