#!/usr/bin/env python

import sys, re, logging
log = logging.getLogger(__name__)


class Item(object):
    @classmethod
    def match(cls, line):
        m = cls.ptn.match(line)
        d = m.groupdict() if not m is None else None
        return None if d is None else cls(d, line) 
    def __init__(self, d, line):
        self.d = d 
        self.line = line
    def __str__(self):
        return self.tmpl % self.d 

class Bullet(Item):
    ptn = re.compile("\* \*\*(?P<bullet>.*)\*\*$")
    tmpl = r"""
    * %(bullet)s 
    """.rstrip()

class Title(Item):
    ptn = re.compile("(?P<year>20\d\d)\s(?P<month>\S*)\s:\s(?P<title>.*)\s*$")
    tmpl = r"""
    %(year)s %(month)s : %(title)s
    """.rstrip()


class Progress(object):
    def __init__(self, lines):
        items = []
        for line in lines:
            title = Title.match(line)
            bullet = Bullet.match(line)
            if title is not None:
                items.append(title)
            elif bullet is not None:
                items.append(bullet)
            else:
                pass
            pass
        pass
        self.items = items
    def __str__(self):
        return "\n".join( map(str, self.items))

   
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    progress = Progress( sys.stdin.readlines() )
    print progress



