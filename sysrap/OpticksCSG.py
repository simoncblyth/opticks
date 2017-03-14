# generated Tue Mar 14 18:17:52 2017 
# from /Users/blyth/opticks/sysrap 
# base OpticksCSG.h stem OpticksCSG 
# with command :  /Users/blyth/opticks/bin/c_enums_to_python.py OpticksCSG.h 
import sys
#0
class CSG_(object):
    ZERO = 0
    UNION = 1
    INTERSECTION = 2
    DIFFERENCE = 3
    PARTLIST = 4
    SPHERE = 5
    BOX = 6
    ZSPHERE = 7
    ZLENS = 8
    PMT = 9
    PRISM = 10
    TUBS = 11
    UNDEFINED = 12

    @classmethod
    def enum(cls):
        return filter(lambda kv:type(kv[1]) is int,cls.__dict__.items())

    @classmethod
    def desc(cls, typ):
        kvs = filter(lambda kv:kv[1] is typ, cls.enum())
        return kvs[0][0] if len(kvs) == 1 else "UNKNOWN"


