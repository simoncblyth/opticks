#pragma once

enum {
     ERROR_LHS_POP_EMPTY         = 0x1 << 0, 
     ERROR_RHS_POP_EMPTY         = 0x1 << 1, 
     ERROR_LHS_END_NONEMPTY      = 0x1 << 2, 
     ERROR_RHS_END_EMPTY         = 0x1 << 3,
     ERROR_BAD_CTRL              = 0x1 << 4,
     ERROR_LHS_OVERFLOW          = 0x1 << 5,
     ERROR_RHS_OVERFLOW          = 0x1 << 6,
     ERROR_LHS_TRANCHE_OVERFLOW  = 0x1 << 7,
     ERROR_RHS_TRANCHE_OVERFLOW  = 0x1 << 8,
     ERROR_RESULT_OVERFLOW       = 0x1 << 9,
     ERROR_OVERFLOW              = 0x1 << 10,
     ERROR_TRANCHE_OVERFLOW      = 0x1 << 11,
     ERROR_POP_EMPTY             = 0x1 << 12,
     ERROR_XOR_SIDE              = 0x1 << 13,
     ERROR_END_EMPTY             = 0x1 << 14,
     ERROR_ROOT_STATE            = 0x1 << 15
};


