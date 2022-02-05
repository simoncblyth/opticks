# generated Sat Feb  5 11:29:17 2022 
# from /Users/blyth/opticks/sysrap 
# base OpticksCSG.h stem OpticksCSG 
# with command :  /Users/blyth/opticks/bin/c_enums_to_python.py OpticksCSG.h 
#0
class CSG_(object):
    ZERO = 0
    TREE = 1
    UNION = 2
    INTERSECTION = 3
    DIFFERENCE = 4
    NODE = 10
    CONTIGUOUS = 11
    DISCONTIGUOUS = 12
    LEAF = 100
    SPHERE = 101
    BOX = 102
    ZSPHERE = 103
    TUBS = 104
    CYLINDER = 105
    SLAB = 106
    PLANE = 107
    CONE = 108
    MULTICONE = 109
    BOX3 = 110
    TRAPEZOID = 111
    CONVEXPOLYHEDRON = 112
    DISC = 113
    SEGMENT = 114
    ELLIPSOID = 115
    TORUS = 116
    HYPERBOLOID = 117
    CUBIC = 118
    INFCYLINDER = 119
    PHICUT = 120
    THETACUT = 121
    UNDEFINED = 122
    OBSOLETE = 1000
    PARTLIST = 1001
    FLAGPARTLIST = 1002
    FLAGNODETREE = 1003
    FLAGINVISIBLE = 1004
    PMT = 1005
    ZLENS = 1006
    PRISM = 1007
    LAST = 1008
    D2V={'zero': 0, 'tree': 1, 'intersection': 3, 'union': 2, 'difference': 4, 'node': 10, 'contiguous': 11, 'discontiguous': 12, 'leaf': 100, 'sphere': 101, 'box': 102, 'zsphere': 103, 'tubs': 104, 'cylinder': 105, 'slab': 106, 'plane': 107, 'cone': 108, 'multicone': 109, 'box3': 110, 'trapezoid': 111, 'convexpolyhedron': 112, 'disc': 113, 'segment': 114, 'ellipsoid': 115, 'torus': 116, 'hyperboloid': 117, 'cubic': 118, 'infcylinder': 119, 'phicut': 120, 'thetacut': 121, 'undefined': 122, 'obsolete': 1000, 'partlist': 1001, 'flagpartlist': 1002, 'flagnodetree': 1003, 'flaginvisible': 1004, 'pmt': 1005, 'zlens': 1006, 'prism': 1007, 'last': 1008}


    @classmethod
    def raw_enum(cls):
        return list(filter(lambda kv:type(kv[1]) is int,cls.__dict__.items()))

    @classmethod
    def enum(cls):
        return cls.D2V.items() if len(cls.D2V) > 0 else cls.raw_enum()

    @classmethod
    def desc(cls, typ):
        kvs = list(filter(lambda kv:kv[1] == typ, cls.enum()))
        return kvs[0][0] if len(kvs) == 1 else "UNKNOWN"

    @classmethod
    def descmask(cls, typ):
        kvs = list(filter(lambda kv:kv[1] & typ, cls.enum())) 
        return ",".join(map(lambda kv:kv[0], kvs))

    @classmethod
    def fromdesc(cls, label):
        kvs = list(filter(lambda kv:kv[0] == label, cls.enum()))
        return kvs[0][1] if len(kvs) == 1 else -1



