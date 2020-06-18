#!/usr/bin/env python
"""
ed.py : content insertion editor
==================================

Inserts the contents of multiple files into one at positions
located by strings at the starts of the lines before or after 
which the insertions are to be done.

::

   ./ed.py /tmp/tt/.bash_profile \
           -l "before:P()"   \
           -i /tmp/tt/p  \  
           -l "after:Q()" \
           -i /tmp/tt/q  \
           -o /tmp/tt/.bash_profile_edited  

   ./ed.py /tmp/tt/.bash_profile -l "before:P()" -i /tmp/tt/p -l "after:Q()" -i /tmp/tt/q -o /tmp/tt/.bash_profile_edited 

* positional argument is the src file to be edited
* number of "-l,--line" options must match the number of "-i,--insert-path" options
* linespec argument to -l must start with "before:" or "after:" followed by the linestart to match
* when -o,--outpath option is provided the edited file is written there

"""

from __future__ import print_function
import sys, re, os, logging, argparse
log = logging.getLogger(__name__)

class Path(object):

   @classmethod
   def Create(cls, path):
       return cls(path) if os.path.exists(path) else None

   def __init__(self, path, empty=False):
       self.path = os.path.expandvars(os.path.expanduser(path)) 
       self.lines = [] if empty else map(lambda s:s.rstrip("\n"), open(self.path,"r").readlines())

   def save(self, path=None):
       if path is None:
           path = self.path 
       pass
       log.info("writing to %s " % path) 
       open(path, "w").write("\n".join(self.lines+[""]))   # avoid chopping tail newline

   def __str__(self):
       return "\n".join(self.lines)

   def __repr__(self):
       return "Ed %s : %d lines " % (self.path, len(self.lines))

   def find_all_linepos(self, linespec):
       before = linespec.startswith("before:")
       after = linespec.startswith("after:")
       assert before ^ after, (before, after)
       linestart = linespec[len("before:"):] if before else linespec[len("after:"):]

       pp = []
       for i,l in enumerate(self.lines):
           if l.startswith(linestart):
               pp.append(i if before else i+1)
           pass 
       return pp

   def find_one_linepos(self, linespec ):
       pp = self.find_all_linepos(linespec)
       pos = -1
       if len(pp) == 0:
          log.info("linespec [%s] not found in %d lines " % (linespec, len(self.lines))) 
       elif len(pp) > 1:
          log.info("linespec [%s] matches multiple positions : %s in the %d lines " % (linespec, len(pp), len(self.lines))) 
       elif len(pp) == 1:
          pos = pp[0]
          log.info("linespec [%s] matches once at position : %s in the %d lines " % (linespec, pos, len(self.lines))) 
       pass
       return pos


class Ed(object):

    @classmethod
    def parse_args(cls, doc):
        parser = argparse.ArgumentParser(doc)
        parser.add_argument( "path", nargs="?", default="~/.bash_profile", help="path to file to be edited" ) 
        parser.add_argument( "-o", "--outpath", default=None, help="path to write the edited file" ) 
        parser.add_argument( "-l", "--line", action="append", default=[], help="before: or after: followed by precise start of the line before/after which text will be inserted" )
        parser.add_argument( "-i", "--insert-path", action="append", default=[], help="path to file containg text to be inserted" )
        parser.add_argument( "--level", default="info", help="logging level" ) 
        args = parser.parse_args()

        fmt = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
        logging.basicConfig(level=getattr(logging,args.level.upper()), format=fmt)

        nargmatch = len(args.line) == len(args.insert_path)
        if not nargmatch:
           log.fatal("the number of -l,--line options must match the number of -i,--insert-path options %d %d " % (len(args.line), len(args.insert_path)))
        assert nargmatch, (len(args.line), len(args.insert_path))

        return args

    def __init__(self, args):
        """ 
        """
        src = Path(args.path)
        log.info("line %r " % args.line)
        log.info("insert_path %r " % args.insert_path)

        before = self.get_before(src, args.line)
        inserts = self.get_inserts(src, args.insert_path)

        dst = Ed.Merge(src, before, inserts)
        self.dst = dst

    def get_before(self, src, line):
        """
        :param src: Path instance of main file to be edited
        :param line: list of strings with linespec in the main file
        :return before: list of before_line integer positions in the main file 
        """
        before = map(lambda b:src.find_one_linepos(b), line)
        f_before = filter(lambda b:b > -1, before)
        log.info("before %r " % before)
        before_ok = len(f_before) == len(before) 

        if not before_ok:
           log.fatal("failed to find all the before-line in the main %s " % src.path )  
        assert before_ok, (f_before,before)
        return before

    def get_inserts(self, src, insert_path):
        """
        :param src: Path instance of main file to be edited
        :param insert_path: list of paths of the text to be inserted
        :return inserts: list of Path instances containing the text to be inserted 
        """
        inserts = map(lambda p:Path.Create(p), insert_path)   
        f_inserts = filter(None, inserts)

        inserts_ok = len(inserts) == len(f_inserts)
        if not inserts_ok:
           log.fatal("failed to find all the insert files %s " % insert_path )  
        assert inserts_ok, (inserts,f_inserts)
        return inserts

    @classmethod 
    def Merge(cls, src, beforepos, inserts):
        """
        Merge the inserts into the main 
        """
        assert len(beforepos) == len(inserts)
        assert sorted(beforepos) == beforepos, "beforepos line numbers must be in ascending order"

        dst = Path(src.path + "_ed", empty=True)
        srcpos = 0 
        for i in range(len(beforepos)):
            bpos = beforepos[i]                        # line number before which to insert
            dst.lines.extend(src.lines[srcpos:bpos])   # copy lines prior to insertion i  
            srcpos = bpos                              # keep track of the position in the src 
            dst.lines.extend(inserts[i].lines)         # copy lines of the insertion
        pass
        dst.lines.extend(src.lines[srcpos:])           # copy lines from src after the last insertion 
        return dst 


if __name__ == '__main__':

   args = Ed.parse_args(__doc__)
   ed = Ed(args)
   #print(ed.dst)
   if not args.outpath is None:
       ed.dst.save(args.outpath)
   pass

