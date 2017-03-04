# generated Sat Mar  4 19:38:01 2017 
# from /Users/blyth/opticks/optixrap/cu 
# with command :  /Users/blyth/opticks/bin/c_enums_to_python.py boolean-solid.h 

# 0 

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


# 1 

CTRL_RETURN_MISS    = 0
CTRL_RETURN_A       = 1
CTRL_RETURN_B       = 2
CTRL_RETURN_FLIP_B  = 3
CTRL_LOOP_A         = 4
CTRL_LOOP_B         = 5
CTRL_UNDEFINED      = 6


# 2 

ERROR_LHS_POP_EMPTY         = 0x1 << 0
ERROR_RHS_POP_EMPTY         = 0x1 << 1
ERROR_LHS_END_NONEMPTY      = 0x1 << 2
ERROR_RHS_END_EMPTY         = 0x1 << 3
ERROR_BAD_CTRL              = 0x1 << 4
ERROR_LHS_OVERFLOW          = 0x1 << 5
ERROR_RHS_OVERFLOW          = 0x1 << 6
ERROR_LHS_TRANCHE_OVERFLOW  = 0x1 << 7
ERROR_RHS_TRANCHE_OVERFLOW  = 0x1 << 8


# 3 

Union_EnterA_EnterB = ReturnAIfCloser | ReturnBIfCloser
Union_EnterA_ExitB  = ReturnBIfCloser | AdvanceAAndLoop
Union_EnterA_MissB  = ReturnA
Union_ExitA_EnterB  = ReturnAIfCloser | AdvanceBAndLoop
Union_ExitA_ExitB   = ReturnAIfFarther | ReturnBIfFarther
Union_ExitA_MissB   = ReturnA 
Union_MissA_EnterB  = ReturnB 
Union_MissA_ExitB   = ReturnB 
Union_MissA_MissB   = ReturnMiss


# 4 

ACloser_Union_EnterA_EnterB = CTRL_RETURN_A
ACloser_Union_EnterA_ExitB  = CTRL_LOOP_A
ACloser_Union_EnterA_MissB  = CTRL_RETURN_A
ACloser_Union_ExitA_EnterB  = CTRL_RETURN_A
ACloser_Union_ExitA_ExitB   = CTRL_RETURN_B
ACloser_Union_ExitA_MissB   = CTRL_RETURN_A
ACloser_Union_MissA_EnterB  = CTRL_RETURN_B
ACloser_Union_MissA_ExitB   = CTRL_RETURN_B
ACloser_Union_MissA_MissB   = CTRL_RETURN_MISS


# 5 

BCloser_Union_EnterA_EnterB = CTRL_RETURN_B
BCloser_Union_EnterA_ExitB  = CTRL_RETURN_B
BCloser_Union_EnterA_MissB  = CTRL_RETURN_A
BCloser_Union_ExitA_EnterB  = CTRL_LOOP_B
BCloser_Union_ExitA_ExitB   = CTRL_RETURN_A
BCloser_Union_ExitA_MissB   = CTRL_RETURN_A
BCloser_Union_MissA_EnterB  = CTRL_RETURN_B
BCloser_Union_MissA_ExitB   = CTRL_RETURN_B
BCloser_Union_MissA_MissB   = CTRL_RETURN_MISS


# 6 

Difference_EnterA_EnterB =  ReturnAIfCloser | AdvanceBAndLoop
Difference_EnterA_ExitB  =  AdvanceAAndLoopIfCloser | AdvanceBAndLoopIfCloser
Difference_EnterA_MissB  =  ReturnA
Difference_ExitA_EnterB  =  ReturnAIfCloser | ReturnFlipBIfCloser
Difference_ExitA_ExitB   =  ReturnFlipBIfCloser | AdvanceAAndLoop
Difference_ExitA_MissB   =  ReturnA
Difference_MissA_EnterB  =  ReturnMiss
Difference_MissA_ExitB   =  ReturnMiss
Difference_MissA_MissB   =  ReturnMiss


# 7 

ACloser_Difference_EnterA_EnterB = CTRL_RETURN_A
ACloser_Difference_EnterA_ExitB  = CTRL_LOOP_A
ACloser_Difference_EnterA_MissB  = CTRL_RETURN_A
ACloser_Difference_ExitA_EnterB  = CTRL_RETURN_A
ACloser_Difference_ExitA_ExitB   = CTRL_LOOP_A
ACloser_Difference_ExitA_MissB   = CTRL_RETURN_A
ACloser_Difference_MissA_EnterB  = CTRL_RETURN_MISS
ACloser_Difference_MissA_ExitB   = CTRL_RETURN_MISS
ACloser_Difference_MissA_MissB   = CTRL_RETURN_MISS



# 8 

BCloser_Difference_EnterA_EnterB = CTRL_LOOP_B
BCloser_Difference_EnterA_ExitB  = CTRL_LOOP_B
BCloser_Difference_EnterA_MissB  = CTRL_RETURN_A
BCloser_Difference_ExitA_EnterB  = CTRL_RETURN_FLIP_B
BCloser_Difference_ExitA_ExitB   = CTRL_RETURN_FLIP_B
BCloser_Difference_ExitA_MissB   = CTRL_RETURN_A
BCloser_Difference_MissA_EnterB  = CTRL_RETURN_MISS
BCloser_Difference_MissA_ExitB   = CTRL_RETURN_MISS
BCloser_Difference_MissA_MissB   = CTRL_RETURN_MISS


# 9 

Intersection_EnterA_EnterB = AdvanceAAndLoopIfCloser | AdvanceBAndLoopIfCloser
Intersection_EnterA_ExitB  = ReturnAIfCloser | AdvanceBAndLoop
Intersection_EnterA_MissB  = ReturnMiss
Intersection_ExitA_EnterB  = ReturnBIfCloser | AdvanceAAndLoop
Intersection_ExitA_ExitB   = ReturnAIfCloser | ReturnBIfCloser
Intersection_ExitA_MissB   = ReturnMiss
Intersection_MissA_EnterB  = ReturnMiss
Intersection_MissA_ExitB   = ReturnMiss
Intersection_MissA_MissB   = ReturnMiss


# 10 

ACloser_Intersection_EnterA_EnterB = CTRL_LOOP_A
ACloser_Intersection_EnterA_ExitB  = CTRL_RETURN_A
ACloser_Intersection_EnterA_MissB  = CTRL_RETURN_MISS
ACloser_Intersection_ExitA_EnterB  = CTRL_LOOP_A
ACloser_Intersection_ExitA_ExitB   = CTRL_RETURN_A
ACloser_Intersection_ExitA_MissB   = CTRL_RETURN_MISS
ACloser_Intersection_MissA_EnterB  = CTRL_RETURN_MISS
ACloser_Intersection_MissA_ExitB   = CTRL_RETURN_MISS
ACloser_Intersection_MissA_MissB   = CTRL_RETURN_MISS


# 11 

BCloser_Intersection_EnterA_EnterB = CTRL_LOOP_B
BCloser_Intersection_EnterA_ExitB  = CTRL_LOOP_B
BCloser_Intersection_EnterA_MissB  = CTRL_RETURN_MISS
BCloser_Intersection_ExitA_EnterB  = CTRL_RETURN_B
BCloser_Intersection_ExitA_ExitB   = CTRL_RETURN_B
BCloser_Intersection_ExitA_MissB   = CTRL_RETURN_MISS
BCloser_Intersection_MissA_EnterB  = CTRL_RETURN_MISS
BCloser_Intersection_MissA_ExitB   = CTRL_RETURN_MISS
BCloser_Intersection_MissA_MissB   = CTRL_RETURN_MISS


# 12 

Enter = 0
Exit  = 1
Miss  = 2

