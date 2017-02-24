#!/usr/bin/env python

from intersect import DIFFERENCE, UNION, INTERSECTION, desc

# intersect status
Enter = 1
Exit  = 2
Miss  = 3
desc_state = { Enter : "Enter", Exit : "Exit", Miss : "Miss" }

# acts
NONE          = 0
RetMiss       = 0x1 << 0 
RetL          = 0x1 << 1 
RetR          = 0x1 << 2 
RetLIfCloser  = 0x1 << 3 
RetRIfCloser  = 0x1 << 4
LoopL         = 0x1 << 5 
LoopLIfCloser = 0x1 << 6
LoopR         = 0x1 << 7 
LoopRIfCloser = 0x1 << 8
FlipR         = 0x1 << 9
BooleanStart  = 0x1 << 10
BooleanError  = 0x1 << 11

ResumeFromLoopL = 0x1 << 12
ResumeFromLoopR = 0x1 << 13
NewTranche      = 0x1 << 14



# RetMiss, RetL, RetR, RetLIfCloser, RetRIfCloser, LoopL, LoopLIfCloser, LoopR, LoopRIfCloser, FlipR


def desc_acts(acts):
    s = ""
    if acts & RetMiss:      s+= "RetMiss "
    if acts & RetL:         s+= "RetL "
    if acts & RetR:         s+= "RetR "
    if acts & RetLIfCloser: s+= "RetLIfCloser "
    if acts & RetRIfCloser: s+= "RetRIfCloser "
    if acts & LoopL:        s+= "LoopL "
    if acts & LoopLIfCloser:s+= "LoopLIfCloser "
    if acts & LoopR:        s+= "LoopR "
    if acts & LoopRIfCloser:s+= "LoopRIfCloser "
    if acts & FlipR:        s+= "FlipR "
    if acts & BooleanStart: s+= "BooleanStart"
    if acts & BooleanError: s+= "BooleanError"
    if acts & ResumeFromLoopL: s+= "ResumeFromLoopL "
    if acts & ResumeFromLoopR: s+= "ResumeFromLoopR "
    if acts & NewTranche: s+= "NewTranche "

    return s 


#
# note that although two loopers do appear together "LoopLIfCloser | LoopRIfCloser" 
# they are always conditionals on which is closer so only one of them will be enacted 
#

table_ = {
    DIFFERENCE : { 
                  Enter : {
                             Enter : RetLIfCloser | LoopR,
                              Exit : LoopLIfCloser | LoopRIfCloser,
                              Miss : RetL
                          },

                  Exit: {
                             Enter : RetLIfCloser | RetRIfCloser | FlipR,
                             Exit  : RetRIfCloser | FlipR | LoopL,
                             Miss  : RetL
                       },

                  Miss: {
                             Enter : RetMiss,
                             Exit : RetMiss,
                             Miss : RetMiss
                        }
               }, 

    UNION : {
                 Enter : {
                            Enter : RetLIfCloser | RetRIfCloser, 
                            Exit  : RetRIfCloser | LoopL,
                            Miss  : RetL
                         },

                 Exit  : {
                            Enter : RetLIfCloser | LoopR, 
                            Exit  : LoopLIfCloser | LoopRIfCloser,
                            Miss  : RetL
                         },

                  Miss: {
                             Enter : RetR,
                             Exit  : RetR,
                             Miss  : RetMiss
                        }
               },
 
   INTERSECTION : {
                         Enter : {
                                    Enter : LoopLIfCloser | LoopRIfCloser,
                                    Exit  : RetLIfCloser | LoopR ,
                                    Miss  : RetMiss
                                 },

                         Exit :  {
                                    Enter : RetRIfCloser | LoopL,
                                    Exit  : RetLIfCloser | RetRIfCloser,
                                    Miss  : RetMiss 
                                 },

                         Miss :  {
                                    Enter : RetMiss,  
                                    Exit  : RetMiss,  
                                    Miss  : RetMiss
                                 }
                      }
}

def boolean_table(operation, l, r):
    assert operation in [UNION, INTERSECTION, DIFFERENCE], operation
    assert l in [Enter,Exit,Miss], l
    assert r in [Enter,Exit,Miss], r
    return table_[operation][l][r]


if __name__ == '__main__':
    for op in [UNION,INTERSECTION,DIFFERENCE]:
        print desc[op]
        for l in [Enter, Exit, Miss]:
            for r in [Enter, Exit, Miss]:
                acts = boolean_table(op, l, r )

                print "   %7s %7s   ->   %35s " % ( desc_state[l], desc_state[r], desc_acts(acts) )
 






