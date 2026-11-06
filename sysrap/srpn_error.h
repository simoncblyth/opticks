#pragma once
/**
srpn_error.h
=============

After CSG/csg_error.h

**/


enum {
     SRPN_ERROR_LHS_POP_EMPTY         = 0x1 << 0,
     SRPN_ERROR_RHS_POP_EMPTY         = 0x1 << 1,
     SRPN_ERROR_LHS_END_NONEMPTY      = 0x1 << 2,
     SRPN_ERROR_RHS_END_EMPTY         = 0x1 << 3,
     SRPN_ERROR_BAD_CTRL              = 0x1 << 4,
     SRPN_ERROR_LHS_OVERFLOW          = 0x1 << 5,
     SRPN_ERROR_RHS_OVERFLOW          = 0x1 << 6,
     SRPN_ERROR_LHS_TRANCHE_OVERFLOW  = 0x1 << 7,
     SRPN_ERROR_RHS_TRANCHE_OVERFLOW  = 0x1 << 8,
     SRPN_ERROR_RESULT_OVERFLOW       = 0x1 << 9,
     SRPN_ERROR_OVERFLOW              = 0x1 << 10,
     SRPN_ERROR_TRANCHE_OVERFLOW      = 0x1 << 11,
     SRPN_ERROR_POP_EMPTY             = 0x1 << 12,
     SRPN_ERROR_XOR_SIDE              = 0x1 << 13,
     SRPN_ERROR_END_EMPTY             = 0x1 << 14,
     SRPN_ERROR_ROOT_STATE            = 0x1 << 15
};


