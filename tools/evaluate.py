#!/usr/bin/python

from collections import OrderedDict

def evaluate_var(v, vdump=False, error=None):
    """
    :param v: SBValue 
    :return: python equivalent or "?" if unhandled
    """
    fmt = " %(k)10s : %(tn)30s : %(nc)4d : %(sz)4d : %(e)15s : %(te)10s :  %(v)40s "


    nc = v.GetNumChildren()
    k = v.GetName()
    t = v.GetType()
    sz = v.GetByteSize()
    tn = v.GetTypeName()   
    d = v.GetData()

    if tn == "unsigned int":
        assert sz == 4
        e = v.GetValueAsUnsigned()
    elif tn == "int":
        assert sz == 4
        e = v.GetValueAsSigned()
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
        offset = 1 
        e = d.GetString(error, offset)  
        # offset 1 avoids "\x16hello"
        # hmm kinda dodgy, the string is actually composite with one child 
    elif nc > 0:
        e = evaluate_obj(v, error=error)  
    else:
        e = "?"
    pass
    te = str(type(e))
    if vdump:
        print fmt % locals()
    pass
    return e


def evaluate_obj(o, odump=False, vdump=False, error=None):
    eo = OrderedDict()
    nc = o.GetNumChildren()
    for i in range(nc):
        v = o.GetChildAtIndex(i)
        k = v.GetName()
        e = evaluate_var(v, vdump, error=error)
        eo[k] = e 
        if odump:
            te = type(e)
            print " %(k)10s : %(e)15s : %(te)15s " %  locals()
        pass
    pass 
    return eo


def evaluate_frame(f, fdump=False, vdump=False, error=None):
    ef = OrderedDict()
    vls = f.get_all_variables()
    for v in vls:
        k = v.GetName()
        e = evaluate_var(v, vdump, error=error)
        ef[k] = e 
        if fdump:
            te = type(e)
            print " %(k)10s : %(e)15s : %(te)15s " %  locals()
        pass
    pass
    return ef



if __name__ == '__main__':
    pass

