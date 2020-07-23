#!/usr/bin/env python

import sys, os, argparse, logging
log = logging.getLogger(__name__)

class Cat(object):
    @classmethod
    def parse_args(cls, doc, **kwa):
        #np.set_printoptions(suppress=True, precision=3 )
        parser = argparse.ArgumentParser(doc)
        parser.add_argument(     "path",  nargs="?", help="Path to a file to cat", default=kwa.get("path",None) )
        parser.add_argument(     "--level", default="info", help="logging level" ) 
        parser.add_argument(     "-s","--selection", default="", help="comma delimited list of line indices, 0-based" ) 
        parser.add_argument(     "-d","--dupes", default=False, action="store_true", help="Count ocurrence of each line" ) 
        args = parser.parse_args()
        fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
        logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)
        return args  

    def __init__(self, path):
        assert os.path.exists(path)
        lines = map(str.strip,open(path, "r").readlines())

        self.path = path 
        self.lines = lines
        self._selection = range(len(lines))
        self.dupes = False        

    def _set_selection(self, q):
        if len(q) == 0: return
        self._selection = map(int, q.split(","))
    def _get_selection(self):
        return self._selection
    selection = property(_get_selection, _set_selection)

    def __str__(self):
        lines = self.lines
        if self.dupes:
            dupes_ = lambda line:len(filter(lambda _:_ == line, lines))
            return "\n".join(["%2s %-4d %-4d %s" % (dupes_(lines[i]), i,i+1, lines[i]) for i in self.selection])
        else:
            return "\n".join(["%-4d %-4d %s" % (i,i+1, lines[i]) for i in self.selection])
        pass 

if __name__ == '__main__':
    args = Cat.parse_args(__doc__)
    path = args.path 

    cat = Cat(path)
    cat.selection = args.selection 
    cat.dupes = args.dupes
    print(cat)



