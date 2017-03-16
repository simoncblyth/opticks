#!/usr/bin/env python
"""
Usage example::

    oxrap-;oxrap-cd cu
    c_enums_to_python.py boolean_solid.h # check 
    c_enums_to_python.py boolean_solid.h > boolean_solid.py 

    sysrap-;sysrap-cd 
    c_enums_to_python.py OpticksCSG.h  # check 
    c_enums_to_python.py OpticksCSG.h > OpticksCSG.py 




"""
import sys, datetime, os, logging
log = logging.getLogger(__name__)

indent_ = lambda lines:"\n".join(["%s%s" % (" " * 4, line) for line in lines])
trim_ = lambda txt:txt.strip().rstrip()
unquote_ = lambda txt:txt.replace("\"","")
first_ = lambda kv:kv[0]
second_ = lambda kv:kv[1]


class Strings(object):
    """
    Parse header enum string consts such as the below into a dict::

        static const char* CSG_ZERO_          = "zero" ; 
        static const char* CSG_INTERSECTION_  = "intersection" ; 
        static const char* CSG_UNION_         = "union" ; 
        static const char* CSG_DIFFERENCE_    = "difference" ; 
        static const char* CSG_PARTLIST_      = "partlist" ; 

    """
    pfx = "static const char* "
    def __init__(self, txt):
        lines = filter(lambda line:line.find("=")>-1 and line.find(";")>-1,filter(lambda line:line.startswith(self.pfx), txt.splitlines()))
        lines = map(lambda line:line[len(self.pfx):line.index(";")], lines)
        kvs = map(lambda line:map(unquote_,map(trim_,line.split("="))), lines)
        self.kvs = kvs 

    def getkv(self, kpfx):
        fkvs = filter(lambda kv:kv[0].startswith(kpfx), self.kvs)
        keys = map(lambda k:k[:-1],filter(lambda k:k.endswith("_"),map(lambda k:k[len(kpfx):],map(trim_,map(first_,  fkvs)))))
        vals = map(trim_,map(second_, fkvs))

        if len(keys) != len(vals):
            log.fatal("enum string keys are expected to end with an _")
            assert 0  

        return keys, vals

    def __repr__(self):
        return repr(self.kvs)
 

class Enum(object):

    tail_template = r"""

    @classmethod
    def raw_enum(cls):
        return filter(lambda kv:type(kv[1]) is int,cls.__dict__.items())

    @classmethod
    def enum(cls):
        return cls.D2V.items() if len(cls.D2V) > 0 else cls.raw_enum()

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

    @classmethod
    def has_curlies(cls, eraw):
        return eraw.find("{") > -1 and eraw.find("}") > -1  

    def __init__(self, eraw, index, hdr):
        """
        :param eraw:C enum source text
        """
        assert self.has_curlies(eraw)
        self.index = index
        self.hdr = hdr
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
        self.keys = map(lambda k:k[len(kpfx):], keys)
        self.vals = map(_translate_val,vals)
        self.kmap = kmap

    def head(self):
        return "\n".join(filter(None,["#%d" % self.index, "class %s(object):" % self.kls if self.kls is not None else None])) 

    def body(self):
        lines = map(lambda i:"%s = %s" % (self.keys[i], self.vals[i]), range(len(self.keys))) 

        sk,sv = self.hdr.strings.getkv(self.kpfx)
        vk = map( lambda k:int(self.vals[self.keys.index(k)]), sk)
        #print "keys:%r" % self.keys 

        srep = []
        #srep.append("V2D=%r" % dict(zip(vk,sv)))
        srep.append("D2V=%r" % dict(zip(sv,vk)))
        return "\n".join(map(indent_,[lines, srep]))

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
        self.strings = Strings(txt)

        c_enums = txt.split("enum")[1:]

        self.enums = []
        for i,eraw in enumerate(c_enums):
            if Enum.has_curlies(eraw):  
                self.add(eraw, i)
        pass

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
               ])

    def body(self):
        return "\n".join(map(repr, self.enums))

    def tail(self):
        return ""

    def __repr__(self):
        return "\n".join([self.head(),self.body(),self.tail()])




if __name__ == '__main__':
   logging.basicConfig(level=logging.INFO)
   assert len(sys.argv) > 1
   hdr = Hdr(sys.argv[1], sys.argv[0])
   print hdr
   #print hdr.strings


 


