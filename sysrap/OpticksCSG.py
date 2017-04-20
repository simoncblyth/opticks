# generated Thu Apr 20 16:17:30 2017 
# from /Users/blyth/opticks/sysrap 
# base OpticksCSG.h stem OpticksCSG 
# with command :  /Users/blyth/opticks/bin/c_enums_to_python.py OpticksCSG.h 
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
    CYLINDER = 12
    SLAB = 13
    PLANE = 14
    CONE = 15
    UNDEFINED = 16
    FLAGPARTLIST = 100
    FLAGNODETREE = 101
    D2V={'box': 6, 'zlens': 8, 'cylinder': 12, 'difference': 3, 'undefined': 16, 'union': 1, 'slab': 13, 'pmt': 9, 'plane': 14, 'prism': 10, 'zsphere': 7, 'sphere': 5, 'zero': 0, 'partlist': 4, 'cone': 15, 'intersection': 2, 'tubs': 11, 'flagnodetree': 101, 'flagpartlist': 100}


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



