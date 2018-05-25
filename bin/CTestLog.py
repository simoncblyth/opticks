#!/usr/bin/env python
"""
CTestLog.py
=============

Collective reporting from a bunch of separate ctest.log files::

    CTestLog.py /usr/local/opticks-cmake-overhaul/build

"""
import sys, re, os, logging, argparse, datetime
log = logging.getLogger(__name__)


class Test(dict):

    tmpl = "  %(num)-3s/%(den)-3s Test #%(num2)-3s: %(name)-45s %(result)-30s %(time)-6s "  

    def __init__(self, *args, **kwa):
        dict.__init__(self, *args, **kwa)
    def __repr__(self):
        return self.tmpl % self


class CTestLog(object):
    """
    47/49 Test #47: GGeoTest.RecordsNPYTest ..........   Passed    0.03 sec

    """
    NAME = "ctest.log" 
    TPATN = re.compile("\s*(?P<num>\d*)/(?P<den>\d*)\s*Test\s*#(?P<num2>\d*):\s*(?P<name>\S*)\s*(?P<div>\.*)\s*(?P<result>.*)\s+(?P<time>\d+\.\d+) sec$") 

    @classmethod
    def examine_logs(cls, args):
        logs = []
        root = args.base 
        for dirpath, dirs, names in os.walk(root):
            if cls.NAME in names:
                log.debug(dirpath)
                reldir = dirpath[len(root)+1:]
                if reldir == "" and not args.withtop: 
                    log.debug("skipping toplevel tests, reldir [%s]" % reldir)
                    continue 
                pass
                path = os.path.join(dirpath, cls.NAME)
                lines = map(str.rstrip, file(path,"r").readlines() ) 
                lg = cls(lines, path=path, reldir=reldir)
                logs.append(lg)
            pass
        pass
        tot = {}
        tot["tests"] = 0 
        tot["fails"] = 0 
        for lg in logs:
            tot["tests"] += len(lg.tests) 
            tot["fails"] += len(lg.fails) 
        pass
        cls.logs = logs 
        cls.tot = tot

    @classmethod
    def desc_totals(cls):
        return " totals  %(fails)-3s / %(tests)-3s " % cls.tot

 
    def __init__(self, lines, path=None, reldir=None):
        self.lines = lines
        self.reldir = reldir
        self.name = os.path.basename(reldir)
        self.path = path 
        self.tests = []
        self.fails = []
        dt = datetime.datetime.fromtimestamp(os.stat(path).st_ctime) if path is not None else None
        self.dt = dt 

        for line in lines:
            m = self.TPATN.match(line)
            if m:
                tst = Test(m.groupdict())
                self.tests.append(tst)
                if not tst["result"].strip() == "Passed":
                    self.fails.append(tst)
                pass
                #print line  
                #print tst
            pass

 
    def __repr__(self):
        return "CTestLog : %20s : %3s : %s : %s " % ( self.reldir, len(self.lines), self.dt, self.path  )

    def __str__(self):
        return "\n".join([repr(self)] + self.lines )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument( "base", nargs="*",  help="Dump tree" )
    parser.add_argument( "--withtop", action="store_true", help="Switch on handling of the usually skipped top level test results" )
    args = parser.parse_args()

    if len(args.base) == 0:
        args.base = os.getcwd()
    else: 
        args.base = args.base[0]
    pass

    CTestLog.examine_logs(args)
    print CTestLog.desc_totals()

    for lg in CTestLog.logs:
        for tst in lg.tests:
            print tst
        pass
    pass
    for lg in CTestLog.logs:
        for tst in lg.fails:
            print tst
        pass
    pass


