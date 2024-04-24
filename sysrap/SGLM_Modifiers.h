#pragma once
/**
SGLM_Modifiers.h : control keys enumeration 
============================================

**/

struct SGLM_Modifiers
{
    enum { 
           MOD_SHIFT   = 0x1 << 0,
           MOD_CONTROL = 0x1 << 1,
           MOD_ALT     = 0x1 << 2,
           MOD_SUPER   = 0x1 << 3,
           MOD_W       = 0x1 << 4,
           MOD_A       = 0x1 << 5,
           MOD_S       = 0x1 << 6,
           MOD_D       = 0x1 << 7,
           MOD_Z       = 0x1 << 8,
           MOD_X       = 0x1 << 9,
           MOD_Q       = 0x1 << 10,
           MOD_E       = 0x1 << 11,
           MOD_R       = 0x1 << 12,
           MOD_Y       = 0x1 << 13
          };

    static bool IsShift(unsigned modifiers);
    static bool IsControl(unsigned modifiers);
    static bool IsAlt(unsigned modifiers);
    static bool IsSuper(unsigned modifiers);
    static bool IsW(unsigned modifiers);
    static bool IsA(unsigned modifiers);
    static bool IsS(unsigned modifiers);
    static bool IsD(unsigned modifiers);
    static bool IsZ(unsigned modifiers);
    static bool IsX(unsigned modifiers);
    static bool IsQ(unsigned modifiers);
    static bool IsE(unsigned modifiers);
    static bool IsR(unsigned modifiers);
    static bool IsY(unsigned modifiers);

    static std::string Desc(unsigned modifiers);
};

inline bool SGLM_Modifiers::IsShift(  unsigned modifiers) { return 0 != (modifiers & MOD_SHIFT) ; }
inline bool SGLM_Modifiers::IsControl(unsigned modifiers) { return 0 != (modifiers & MOD_CONTROL) ; }
inline bool SGLM_Modifiers::IsAlt(    unsigned modifiers) { return 0 != (modifiers & MOD_ALT) ; }
inline bool SGLM_Modifiers::IsSuper(  unsigned modifiers) { return 0 != (modifiers & MOD_SUPER) ; }
inline bool SGLM_Modifiers::IsW(      unsigned modifiers) { return 0 != (modifiers & MOD_W) ; }
inline bool SGLM_Modifiers::IsA(      unsigned modifiers) { return 0 != (modifiers & MOD_A) ; }
inline bool SGLM_Modifiers::IsS(      unsigned modifiers) { return 0 != (modifiers & MOD_S) ; }
inline bool SGLM_Modifiers::IsD(      unsigned modifiers) { return 0 != (modifiers & MOD_D) ; }
inline bool SGLM_Modifiers::IsZ(      unsigned modifiers) { return 0 != (modifiers & MOD_Z) ; }
inline bool SGLM_Modifiers::IsX(      unsigned modifiers) { return 0 != (modifiers & MOD_X) ; }
inline bool SGLM_Modifiers::IsQ(      unsigned modifiers) { return 0 != (modifiers & MOD_Q) ; }
inline bool SGLM_Modifiers::IsE(      unsigned modifiers) { return 0 != (modifiers & MOD_E) ; }
inline bool SGLM_Modifiers::IsR(      unsigned modifiers) { return 0 != (modifiers & MOD_R) ; }
inline bool SGLM_Modifiers::IsY(      unsigned modifiers) { return 0 != (modifiers & MOD_Y) ; }

inline std::string SGLM_Modifiers::Desc(unsigned modifiers)
{
    std::stringstream ss ; 
    ss << "SGLM_Modifiers::Desc[" 
       << ( IsShift(  modifiers) ? " SHIFT"   : "" )     
       << ( IsControl(modifiers) ? " CONTROL" : "" )   
       << ( IsAlt(    modifiers) ? " ALT"     : "" ) 
       << ( IsSuper(  modifiers) ? " SUPER"   : "" )
       << ( IsW(      modifiers) ? " W"   : "" )
       << ( IsA(      modifiers) ? " A"   : "" )
       << ( IsS(      modifiers) ? " S"   : "" )
       << ( IsD(      modifiers) ? " D"   : "" )
       << ( IsZ(      modifiers) ? " Z"   : "" )
       << ( IsX(      modifiers) ? " X"   : "" )
       << ( IsQ(      modifiers) ? " Q"   : "" )
       << ( IsE(      modifiers) ? " E"   : "" )
       << ( IsR(      modifiers) ? " R"   : "" )
       << ( IsY(      modifiers) ? " Y"   : "" )
       <<  "]"
       ;
    std::string str = ss.str();
    return str ; 
}


