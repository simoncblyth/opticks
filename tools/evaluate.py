#!/usr/bin/python
"""


NB lldb.frame and frame are not the same 

INTERACTIVE::

    >>> from opticks.tools.evaluate import EV ; self = EV(lldb.frame.FindVariable("this"))
    
WITHIN BREAKPOINT FUNC::

    self = EV(frame.FindVariable("this"))


::

    (lldb) fr v
    (CRandomEngine *) this = 0x0000000110025c70

    (lldb) script
    Python Interactive Interpreter. To exit, type 'quit()', 'exit()' or Ctrl-D.

    >>> from opticks.tools.evaluate import EV ; self = EV(lldb.frame.FindVariable("this"))

    >>> self.v.keys
    ['CLHEP::HepRandomEngine', 'm_g4', 'm_ctx', 'm_ok', 'm_mask', 'm_masked', 'm_path', 'm_alignlevel', 'm_seed', 'm_internal', 'm_skipdupe', 'm_locseq', 'm_curand', 'm_curand_index', 'm_curand_ni', 'm_curand_nv', 'm_current_record_flat_count', 'm_current_step_flat_count', 'm_offset', 'm_offset_count', 'm_flat', 'm_location', 'm_sequence', 'm_cursor', 'm_cursor_old']

    >>> self.v(".m_ctx._step_id")
    (int) _step_id = 0

    >>> self.ev(".m_ctx._step_id")
    0

    >>> self.ev(".m_ctx")
    OrderedDict([('_ok', 'ptr'), ('_pindex', 0), ('_print', True), ('_dbgrec', False), ('_dbgseq', False), ('_dbgzero', False), ('_photons_per_g4event', 10000), ('_steps_per_photon', 10), ('_gen', 4096), ('_record_max', 1), ('_bounce_max', 9), ('_ok_event_init', True), ('_event', 'ptr'), ('_event_id', 0), ('_event_total', 1), ('_event_track_count', 1), ('_track', 'ptr'), ('_process_manager', 'ptr'), ('_track_id', 0), ('_track_total', 1), ('_track_step_count', 1), ('_parent_id', -1), ('_optical', True), ('_pdg_encoding', 0), ('_primary_id', -2), ('_photon_id', 0), ('_reemtrack', False), ('_record_id', 0), ('_record_fraction', 0.0), ('_rejoin_count', 0), ('_primarystep_count', 1), ('_stage', OrderedDict()), ('_debug', False), ('_other', False), ('_dump', False), ('_dump_count', 0), ('_step', 'ptr'), ('_noZeroSteps', 0), ('_step_id', 0), ('_step_total', 1), ('_step_origin', OrderedDict([('dx', 11.291412353515625), ('dy', -34.645111083984375), ('dz', -449.8999938964844)]))])

    >>> self.v(".m_ctx")
    (CG4Ctx &) m_ctx = 0x0000000110025af0: {
      _ok = 0x000000010c735c40
      _pindex = 0
      _print = true
      _dbgrec = false
      _dbgseq = false
      _dbgzero = false
      _photons_per_g4event = 10000
      _steps_per_photon = 10
      _gen = 4096
      _record_max = 1





"""

from collections import OrderedDict
from opticks.tools.lldb_ import lldb

def rsplit(r):
    return map(lambda _:_.lstrip().rstrip(),r.split("\n"))



class EV(object):
    def __init__(self, v=None):
        self.e = Evaluate()
        self.v = v

    def _set_v(self, v):
        if type(v) is Value or type(v) is None:
            self._v = v
        else:
            self._v = Value(v)
        pass 
    def _get_v(self):
        return self._v

    v = property(_get_v, _set_v)

    def v_(self, k):
        if self.v is None: 
            return None
        return self.v(k)

    def ev(self, k):
        if self.v is None: 
            return None
        return self.e(self.v(k))





class Value(object):
    """

    >>> from opticks.tools.evaluate import Value, Evaluate ; e = Evaluate() ; v = Value(lldb.frame.FindVariable("this"))
    >>> v
    (CRandomEngine *) this = 0x000000010f7e2470

    >>> v.keys
    ['CLHEP::HepRandomEngine', 'm_g4', 'm_ctx', 'm_ok', 'm_mask', 'm_masked', 'm_path', 'm_alignlevel', 'm_seed', 'm_internal', 'm_skipdupe', 'm_locseq', 'm_curand', 'm_curand_index', 'm_curand_ni', 'm_curand_nv', 'm_current_record_flat_count', 'm_current_step_flat_count', 'm_flat', 'm_location', 'm_sequence', 'm_cursor']

    >>> v("m_location")
    (std::__1::string) m_location = "OpBoundary;"
    >>> e(v("m_location"))
    'OpBoundary;'

    >>> ef = e.evaluate_frame(lldb.frame)

    """
    def __init__(self, v):
        self.v = v

    def _get_keys(self):        
        nc = self.v.GetNumChildren()
        return [self.v.GetChildAtIndex(i).GetName() for i in range(nc)]
    keys = property(_get_keys)

    def __call__(self, k ):
        if k[0] == ".":
            vv = self.v.GetValueForExpressionPath(k)
        else:  
            vv = self.v.GetChildMemberWithName(k) 
        pass
        return Value(vv)

    def __repr__(self):
        return str(self.v)        

    def __str__(self):
        return "\n".join( map(repr,map(self, self.keys) ))





class Evaluate(object):
    """
    NB : holds no "domain" state
    """
    SKIPS = rsplit(r"""
    char **
    """)

    NOT_CANONICAL = rsplit(r"""
    std::__1::string
    """)
    # canonical type for std::string is giant basic_string monstrosity, so dont use it for classify

    E_ATOM = "ATOM"
    E_SKIP = "SKIP"
    E_PTR = "PTR"
    E_COMP = "COMP"
    E_ENUM = "ENUM"

    @classmethod
    def classify(cls, v):
        tn = v.GetTypeName()           

        t = v.GetType()
        pt = t.IsPointerType()

        lem = len(t.enum_members)

        if tn in cls.ATOMS:
            et = cls.E_ATOM
        elif tn in cls.SKIPS:
            et = cls.E_SKIP
        elif lem > 0:
            et = cls.E_ENUM
        elif pt:
            et = cls.E_PTR 
        else:
            et = cls.E_COMP
        pass
        return et 

    def __init__(self, error=None, opt=""):
        if error is None:
            error = lldb.SBError() 
        pass
        self.error = error
        self.opt = opt

    def __call__(self, v ):
        if type(v) is Value:
            vv = v.v
        else:
            vv = v 
        pass
        return self.evaluate(vv) 

    def evaluate_frame(self, f):
        ef = OrderedDict()
        vls = f.get_all_variables()

        for v in vls:
            k = v.GetName()
            e = self.evaluate(v)
            ef[k] = e
            if "f" in self.opt:
                te = type(e)
                print "(f) %(k)10s : %(e)15s : %(te)15s  " %  locals()
            pass
        pass
        return ef


    def evaluate(self, v):
        et = self.classify(v)

        k = v.GetName()
        nc = v.GetNumChildren()
        tn = v.GetTypeName()   

        if "e" in self.opt:
            print "(e) %(k)10s : %(tn)15s : %(nc)4d : %(et)s " %  locals()
        pass
               
        if et == self.E_ATOM:
            e = self.evaluate_atom(v)
        elif et == self.E_ENUM:
            e = v.GetValue()
        elif et == self.E_SKIP:
            e = "skp"
        elif et == self.E_PTR:
            e = "ptr"
        elif et == self.E_COMP:
            e = self.evaluate_comp(v)
        else:
            assert 0
        pass
        return e


    def evaluate_comp(self, o):
        eo = OrderedDict()
        nc = o.GetNumChildren()

        for i in range(nc):
            v = o.GetChildAtIndex(i)
            k = v.GetName()
            
            eo[k] = self.evaluate(v)

            if "c" in self.opt:
                te = type(e)
                print "(c) %(k)10s : %(e)15s : %(te)15s " %  locals()
            pass
        pass 
        return eo
       


    ATOMS = rsplit(r"""
    bool
    char
    int
    long
    long long
    unsigned char
    unsigned int
    unsigned long
    unsigned long long
    float
    double
    std::__1::string
    const char *
    """)


    def atom_typename(self, v):
        t = v.GetType()
        vtn = v.GetTypeName()   

        if vtn in self.NOT_CANONICAL:
            tn = vtn 
        else: 
            ct = t.GetCanonicalType()
            ctn = ct.GetName() 
            tn = ctn 
        pass
        return tn

 
    def evaluate_atom(self, v):
        """
        :param v: SBValue 
        :return: python equivalent or "?" if unhandled
        """
        nc = v.GetNumChildren()
        k = v.GetName()

        tn = self.atom_typename(v)

        sz = v.GetByteSize()
        d = v.GetData()
        error = self.error

        if tn == "unsigned int":
            assert sz == 4
            e = v.GetValueAsUnsigned()
        elif tn == "int":
            assert sz == 4
            e = v.GetValueAsSigned()
        elif tn == "long" or tn == "long long":
            assert sz == 8
            e = d.GetSignedInt64(error, 0)
        elif tn == "unsigned long" or tn == "unsigned long long":
            assert sz == 8
            e = d.GetUnsignedInt64(error, 0)
        elif tn == "float":
            assert sz == 4
            offset = 0 
            e = d.GetFloat(error, offset)
        elif tn == "double":
            assert sz == 8
            offset = 0 
            e = d.GetDouble(error, offset)
        elif tn == "bool":
            assert sz == 1
            offset = 0 
            e = d.GetUnsignedInt8(error, offset)
            assert e == 0 or e == 1
            e = e == 1 
        elif tn == "unsigned char":
            assert sz == 1
            offset = 0 
            e = d.GetUnsignedInt8(error, offset)
        elif tn == "char":
            assert sz == 1
            offset = 0 
            e = d.GetSignedInt8(error, offset)
        elif tn == "std::__1::string":
            s = v.GetSummary()
            e = s[1:-1]    # unquote

            #offset = 1 
            #e = d.GetString(error, offset)  
            #    offset 1 avoids "\x16hello"
            #    hmm kinda dodgy, the string is actually composite with one child 
            #    sometimes gives blanks
            #
            #e = v.GetFrame().EvaluateExpression("%s.c_str()" % k)

        elif tn == "const char *":

            tt = v.GetType().GetPointeeType()
            assert tt.GetName() == "const char"
            sz = tt.GetByteSize() 
            assert sz == 1
            ptr = v.GetValueAsUnsigned()
            e = v.GetFrame().GetThread().GetProcess().ReadCStringFromMemory(ptr,256, error)

        else:
            e = "?"
        pass
        te = str(type(e))
        fmt = "(a) %(k)10s : %(tn)30s : %(nc)4d : %(sz)4d : %(e)15s : %(te)10s :  %(v)40s "
        if "a" in self.opt:
            print fmt % locals()
        pass
        return e

if __name__ == '__main__':
    pass

