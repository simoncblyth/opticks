#!/usr/bin/env python

CtrlBoth            = 0x1 << 0
CtrlLeft            = 0x1 << 1
CtrlRight           = 0x1 << 2
CtrlResumeFromLeft  = 0x1 << 3
CtrlResumeFromRight = 0x1 << 4
CtrlBreak           = 0x1 << 5
CtrlReturn          = 0x1 << 6

def desc_ctrl(ctrl):
    s = ""
    if ctrl & CtrlBoth: s+= "CtrlBoth "
    if ctrl & CtrlReturn: s+= "CtrlReturn "
    if ctrl & CtrlBreak: s+= "CtrlBreak "
    if ctrl & CtrlLeft: s+= "CtrlLeft "
    if ctrl & CtrlRight: s+= "CtrlRight "
    if ctrl & CtrlResumeFromLeft: s+= "CtrlResumeFromLeft "
    if ctrl & CtrlResumeFromRight: s+= "CtrlResumeFromRight "
    return s




