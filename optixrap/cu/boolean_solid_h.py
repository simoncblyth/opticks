# generated Mon Feb 27 11:03:28 2017 
# from /Users/blyth/opticks/optixrap/cu 
# with command :  /Users/blyth/opticks/bin/c_enums_to_python.py boolean-solid.h 

ReturnMiss              = 0x1 << 0
ReturnAIfCloser         = 0x1 << 1
ReturnAIfFarther        = 0x1 << 2
ReturnA                 = 0x1 << 3
ReturnBIfCloser         = 0x1 << 4
ReturnBIfFarther        = 0x1 << 5
ReturnB                 = 0x1 << 6
ReturnFlipBIfCloser     = 0x1 << 7
AdvanceAAndLoop         = 0x1 << 8
AdvanceBAndLoop         = 0x1 << 9
AdvanceAAndLoopIfCloser = 0x1 << 10
AdvanceBAndLoopIfCloser = 0x1 << 11



Union_EnterA_EnterB = ReturnAIfCloser | ReturnBIfCloser
Union_EnterA_ExitB  = ReturnBIfCloser | AdvanceAAndLoop
Union_EnterA_MissB  = ReturnA
Union_ExitA_EnterB  = ReturnAIfCloser | AdvanceBAndLoop
Union_ExitA_ExitB   = ReturnAIfFarther | ReturnBIfFarther
Union_ExitA_MissB   = ReturnA 
Union_MissA_EnterB  = ReturnB 
Union_MissA_ExitB   = ReturnB 
Union_MissA_MissB   = ReturnMiss



Difference_EnterA_EnterB =  ReturnAIfCloser | AdvanceBAndLoop
Difference_EnterA_ExitB  =  AdvanceAAndLoopIfCloser | AdvanceBAndLoopIfCloser
Difference_EnterA_MissB  =  ReturnA
Difference_ExitA_EnterB  =  ReturnAIfCloser | ReturnFlipBIfCloser
Difference_ExitA_ExitB   =  ReturnFlipBIfCloser | AdvanceAAndLoop
Difference_ExitA_MissB   =  ReturnA
Difference_MissA_EnterB  =  ReturnMiss
Difference_MissA_ExitB   =  ReturnMiss
Difference_MissA_MissB   =  ReturnMiss



Intersection_EnterA_EnterB = AdvanceAAndLoopIfCloser | AdvanceBAndLoopIfCloser
Intersection_EnterA_ExitB  = ReturnAIfCloser | AdvanceBAndLoop
Intersection_EnterA_MissB  = ReturnMiss
Intersection_ExitA_EnterB  = ReturnBIfCloser | AdvanceAAndLoop
Intersection_ExitA_ExitB   = ReturnAIfCloser | ReturnBIfCloser
Intersection_ExitA_MissB   = ReturnMiss
Intersection_MissA_EnterB  = ReturnMiss
Intersection_MissA_ExitB   = ReturnMiss
Intersection_MissA_MissB   = ReturnMiss

