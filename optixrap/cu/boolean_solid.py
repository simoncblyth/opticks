# generated Tue Mar 14 18:57:46 2017 
# from /Users/blyth/opticks/optixrap/cu 
# base boolean-solid.h stem boolean-solid 
# with command :  /Users/blyth/opticks/bin/c_enums_to_python.py boolean-solid.h 
import sys
#0
class Act_(object):
    ReturnMiss = 0x1 << 0
    ReturnAIfCloser = 0x1 << 1
    ReturnAIfFarther = 0x1 << 2
    ReturnA = 0x1 << 3
    ReturnBIfCloser = 0x1 << 4
    ReturnBIfFarther = 0x1 << 5
    ReturnB = 0x1 << 6
    ReturnFlipBIfCloser = 0x1 << 7
    AdvanceAAndLoop = 0x1 << 8
    AdvanceBAndLoop = 0x1 << 9
    AdvanceAAndLoopIfCloser = 0x1 << 10
    AdvanceBAndLoopIfCloser = 0x1 << 11

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


#1
class CTRL_(object):
    RETURN_MISS = 0
    RETURN_A = 1
    RETURN_B = 2
    RETURN_FLIP_B = 3
    LOOP_A = 4
    LOOP_B = 5

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


#2
class State_(object):
    Enter = 0
    Exit = 1
    Miss = 2

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


#3
class ERROR_(object):
    LHS_POP_EMPTY = 0x1 << 0
    RHS_POP_EMPTY = 0x1 << 1
    LHS_END_NONEMPTY = 0x1 << 2
    RHS_END_EMPTY = 0x1 << 3
    BAD_CTRL = 0x1 << 4
    LHS_OVERFLOW = 0x1 << 5
    RHS_OVERFLOW = 0x1 << 6
    LHS_TRANCHE_OVERFLOW = 0x1 << 7
    RHS_TRANCHE_OVERFLOW = 0x1 << 8
    RESULT_OVERFLOW = 0x1 << 9
    OVERFLOW = 0x1 << 10
    TRANCHE_OVERFLOW = 0x1 << 11
    POP_EMPTY = 0x1 << 12
    XOR_SIDE = 0x1 << 13
    END_EMPTY = 0x1 << 14
    ROOT_STATE = 0x1 << 15

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


#4
class Union_(object):
    EnterA_EnterB = Act_.ReturnAIfCloser | Act_.ReturnBIfCloser
    EnterA_ExitB = Act_.ReturnBIfCloser | Act_.AdvanceAAndLoop
    EnterA_MissB = Act_.ReturnA
    ExitA_EnterB = Act_.ReturnAIfCloser | Act_.AdvanceBAndLoop
    ExitA_ExitB = Act_.ReturnAIfFarther | Act_.ReturnBIfFarther
    ExitA_MissB = Act_.ReturnA
    MissA_EnterB = Act_.ReturnB
    MissA_ExitB = Act_.ReturnB
    MissA_MissB = Act_.ReturnMiss

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


#5
class ACloser_Union_(object):
    EnterA_EnterB = CTRL_.RETURN_A
    EnterA_ExitB = CTRL_.LOOP_A
    EnterA_MissB = CTRL_.RETURN_A
    ExitA_EnterB = CTRL_.RETURN_A
    ExitA_ExitB = CTRL_.RETURN_B
    ExitA_MissB = CTRL_.RETURN_A
    MissA_EnterB = CTRL_.RETURN_B
    MissA_ExitB = CTRL_.RETURN_B
    MissA_MissB = CTRL_.RETURN_MISS

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


#6
class BCloser_Union_(object):
    EnterA_EnterB = CTRL_.RETURN_B
    EnterA_ExitB = CTRL_.RETURN_B
    EnterA_MissB = CTRL_.RETURN_A
    ExitA_EnterB = CTRL_.LOOP_B
    ExitA_ExitB = CTRL_.RETURN_A
    ExitA_MissB = CTRL_.RETURN_A
    MissA_EnterB = CTRL_.RETURN_B
    MissA_ExitB = CTRL_.RETURN_B
    MissA_MissB = CTRL_.RETURN_MISS

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


#7
class Difference_(object):
    EnterA_EnterB = Act_.ReturnAIfCloser | Act_.AdvanceBAndLoop
    EnterA_ExitB = Act_.AdvanceAAndLoopIfCloser | Act_.AdvanceBAndLoopIfCloser
    EnterA_MissB = Act_.ReturnA
    ExitA_EnterB = Act_.ReturnAIfCloser | Act_.ReturnFlipBIfCloser
    ExitA_ExitB = Act_.ReturnFlipBIfCloser | Act_.AdvanceAAndLoop
    ExitA_MissB = Act_.ReturnA
    MissA_EnterB = Act_.ReturnMiss
    MissA_ExitB = Act_.ReturnMiss
    MissA_MissB = Act_.ReturnMiss

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


#8
class ACloser_Difference_(object):
    EnterA_EnterB = CTRL_.RETURN_A
    EnterA_ExitB = CTRL_.LOOP_A
    EnterA_MissB = CTRL_.RETURN_A
    ExitA_EnterB = CTRL_.RETURN_A
    ExitA_ExitB = CTRL_.LOOP_A
    ExitA_MissB = CTRL_.RETURN_A
    MissA_EnterB = CTRL_.RETURN_MISS
    MissA_ExitB = CTRL_.RETURN_MISS
    MissA_MissB = CTRL_.RETURN_MISS

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


#9
class BCloser_Difference_(object):
    EnterA_EnterB = CTRL_.LOOP_B
    EnterA_ExitB = CTRL_.LOOP_B
    EnterA_MissB = CTRL_.RETURN_A
    ExitA_EnterB = CTRL_.RETURN_FLIP_B
    ExitA_ExitB = CTRL_.RETURN_FLIP_B
    ExitA_MissB = CTRL_.RETURN_A
    MissA_EnterB = CTRL_.RETURN_MISS
    MissA_ExitB = CTRL_.RETURN_MISS
    MissA_MissB = CTRL_.RETURN_MISS

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


#10
class Intersection_(object):
    EnterA_EnterB = Act_.AdvanceAAndLoopIfCloser | Act_.AdvanceBAndLoopIfCloser
    EnterA_ExitB = Act_.ReturnAIfCloser | Act_.AdvanceBAndLoop
    EnterA_MissB = Act_.ReturnMiss
    ExitA_EnterB = Act_.ReturnBIfCloser | Act_.AdvanceAAndLoop
    ExitA_ExitB = Act_.ReturnAIfCloser | Act_.ReturnBIfCloser
    ExitA_MissB = Act_.ReturnMiss
    MissA_EnterB = Act_.ReturnMiss
    MissA_ExitB = Act_.ReturnMiss
    MissA_MissB = Act_.ReturnMiss

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


#11
class ACloser_Intersection_(object):
    EnterA_EnterB = CTRL_.LOOP_A
    EnterA_ExitB = CTRL_.RETURN_A
    EnterA_MissB = CTRL_.RETURN_MISS
    ExitA_EnterB = CTRL_.LOOP_A
    ExitA_ExitB = CTRL_.RETURN_A
    ExitA_MissB = CTRL_.RETURN_MISS
    MissA_EnterB = CTRL_.RETURN_MISS
    MissA_ExitB = CTRL_.RETURN_MISS
    MissA_MissB = CTRL_.RETURN_MISS

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


#12
class BCloser_Intersection_(object):
    EnterA_EnterB = CTRL_.LOOP_B
    EnterA_ExitB = CTRL_.LOOP_B
    EnterA_MissB = CTRL_.RETURN_MISS
    ExitA_EnterB = CTRL_.RETURN_B
    ExitA_ExitB = CTRL_.RETURN_B
    ExitA_MissB = CTRL_.RETURN_MISS
    MissA_EnterB = CTRL_.RETURN_MISS
    MissA_ExitB = CTRL_.RETURN_MISS
    MissA_MissB = CTRL_.RETURN_MISS

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



