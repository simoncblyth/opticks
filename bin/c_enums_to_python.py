#!/usr/bin/env python
"""
Usage example::

    oxrap-;oxrap-cd cu
    c_enums_to_python.py boolean-solid.h # check 
    c_enums_to_python.py boolean-solid.h > boolean_solid.py 

    sysrap-;sysrap-cd 
    c_enums_to_python.py OpticksCSG.h  # check 
    c_enums_to_python.py OpticksCSG.h > OpticksCSG.py 




"""
import sys, datetime, os

indent_ = lambda lines, indent:"\n".join(["%s%s" % (indent, line) for line in lines])



class StaticConstChar(object):
    def __init__(self, lines):
        self.lines = lines
    def __repr__(self):
        return "\n".join(self.lines)


class Enum(object):

    tail_template = r"""
    @classmethod
    def enum(cls):
        return filter(lambda kv:type(kv[1]) is int,cls.__dict__.items())

    @classmethod
    def desc(cls, typ):
        kvs = filter(lambda kv:kv[1] == typ, cls.enum())
        return kvs[0][0] if len(kvs) == 1 else "UNKNOWN"

    @classmethod
    def descmask(cls, typ):
        kvs = filter(lambda kv:kv[1] & typ, cls.enum())
        return ",".join(map(lambda kv:kv[0], kvs))

    @classmethod
    def fromdesc(cls, label):
        kvs = filter(lambda kv:kv[0] == label, cls.enum())
        return kvs[0][1] if len(kvs) == 1 else -1

"""
    def __init__(self, eraw, index, hdr):
        """
        :param eraw:C enum source text
        """
        self.index = index
        self.hdr = hdr

        trim_ = lambda txt:txt.strip().rstrip()

        etxt = eraw[eraw.index("{"):eraw.index("}")+1] 
        lines = map(lambda l:trim_(l).replace(",",""),etxt[1:-1].split("\n"))
        kvs = filter(lambda kv:len(kv) == 2,map(lambda line:line.split("="), lines))
        keys = map(trim_,map(lambda kv:kv[0], kvs))
        vals = map(trim_,map(lambda kv:kv[1], kvs))
        kpfx = os.path.commonprefix(keys)  

        def _kmap(keys, kpfx):
            return dict(zip(keys, map(lambda k:kpfx+"."+k[len(kpfx):], keys)))

        def _translate_one_val(val):
            return self.hdr.kmap.get(val, val)

        def _translate_val(val):
            """ 
            eg yielding: Act_.ReturnAIfCloser | Act_.ReturnBIfCloser
            """
            if val.find(" ") == -1:
                return _translate_one_val(val)
            else:
                return " ".join(map(_translate_one_val, val.split(" ")))
            pass

        if len(kpfx) == 0:
            kls, indent, kmap = None, "", {}
        else:
            kls, indent, kmap = kpfx, " " * 4, _kmap(keys, kpfx)
        pass
        self.kls = kls
        self.indent = indent
        self.kpfx = kpfx
        self.keys = keys
        self.vals = map(_translate_val,vals)
        self.kmap = kmap

    def head(self):
        return "\n".join(filter(None,["#%d" % self.index, "class %s(object):" % self.kls if self.kls is not None else None])) 

    def body(self):
        lines = map(lambda i:"%s = %s" % (self.keys[i][len(self.kpfx):], self.vals[i]), range(len(self.keys))) 
        return indent_(lines,self.indent)

    def tail(self):
        return self.tail_template if self.kls is not None else "#"

    def __repr__(self):
        return "\n".join([self.head(),self.body(),self.tail()])



class Hdr(object):
    def __init__(self, path, cmd):
        self.path = path
        self.cmd = cmd
        self.kmap = {}
        
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        self.base = base
        self.stem = stem
        self.ctx = {}

        txt = file(path).read()
        c_enums = txt.split("enum")[1:]

        self.enums = []
        for i,eraw in enumerate(c_enums):
            self.add(eraw, i)
        pass

        lines = filter(lambda line:line.find("=")>-1,filter(lambda line:line.startswith("static const char*"), txt.splitlines()))
        self.scc = StaticConstChar(lines)


    def add(self, eraw, i):
        e = Enum(eraw, i, self)
        self.kmap.update(e.kmap)
        self.enums.append(e)

    def head(self):
        now = datetime.datetime.now().strftime("%c")
        return "\n".join([
               "# generated %s " % (now),
               "# from %s " % os.getcwd(),
               "# base %s stem %s " % (self.base, self.stem),
               "# with command :  %s %s " % (self.cmd, self.path),
               "import sys"
               ])

    def body(self):
        return "\n".join(map(repr, self.enums))

    def tail(self):
        return ""

    def __repr__(self):
        return "\n".join([self.head(),self.body(),self.tail()])




if __name__ == '__main__':
   assert len(sys.argv) > 1
   hdr = Hdr(sys.argv[1], sys.argv[0])
   print hdr
   #print hdr.scc


 


