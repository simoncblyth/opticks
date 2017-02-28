#!/usr/bin/env python
"""
Hmm the CtrlReturn* does not need to 
be a mask, but the CtrlLoop* does.
Maybe split.

"""
CtrlReturnMiss      = 0x1 << 1
CtrlReturnLeft      = 0x1 << 2
CtrlReturnRight     = 0x1 << 3
CtrlReturnFlipRight = 0x1 << 4

CtrlLoopLeft        = 0x1 << 5
CtrlLoopRight       = 0x1 << 6

_ctrl_index = { 
      CtrlReturnMiss      : 1, 
      CtrlReturnLeft      : 2, 
      CtrlReturnRight     : 3, 
      CtrlReturnFlipRight : 4, 
      CtrlLoopLeft        : 5, 
      CtrlLoopRight       : 6,
}


_ctrl_color = {
       0:'c', 
       1:'k', 
       2:'m', 
       3:'b', 
       4:'r', 
       5:'y',
       6:'r'
}


def ctrl_index(ctrl):
    return _ctrl_index[ctrl]
    
def desc_ctrl(ctrl):
    s = ""
    if ctrl & CtrlReturnMiss: s+= "CtrlReturnMiss "
    if ctrl & CtrlReturnLeft: s+= "CtrlReturnLeft "
    if ctrl & CtrlReturnRight: s+= "CtrlReturnRight "
    if ctrl & CtrlReturnFlipRight: s+= "CtrlReturnFlipRight "

    if ctrl & CtrlLoopLeft: s+= "CtrlLoopLeft "
    if ctrl & CtrlLoopRight: s+= "CtrlLoopRight "

    return s

def desc_ctrl_cu(seqcu, label=""):
    try:  
        ret = "\n".join([label]+[" %s : %2d %5d : %6d : %s " % (_ctrl_color[index], index, 0x1 << int(index), count, desc_ctrl(0x1 << int(index))) for index, count in seqcu])
    except KeyError:
        ret = repr(seqcu)
    pass
    return ret 
                









