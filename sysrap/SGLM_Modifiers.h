#pragma once
/**
SGLM_Modifiers.h : control keys enumeration 
============================================

**/

struct SGLM_Modifiers
{
    // NB these enum values must not clash with SGLM_Modnav
    enum { 
           MOD_SHIFT   = 0x1 << 0,
           MOD_CONTROL = 0x1 << 1,
           MOD_ALT     = 0x1 << 2,
           MOD_SUPER   = 0x1 << 3, 
           MOD_ALL     = MOD_SHIFT | MOD_CONTROL | MOD_ALT | MOD_SUPER
         };

    static bool IsNone(unsigned modifiers);
    static bool IsShift(unsigned modifiers);
    static bool IsControl(unsigned modifiers);
    static bool IsAlt(unsigned modifiers);
    static bool IsSuper(unsigned modifiers);

    static std::string Desc(unsigned modifiers);
};


inline bool SGLM_Modifiers::IsNone(   unsigned modifiers) { return 0 == (modifiers & MOD_ALL  ) ; }
inline bool SGLM_Modifiers::IsShift(  unsigned modifiers) { return 0 != (modifiers & MOD_SHIFT) ; }
inline bool SGLM_Modifiers::IsControl(unsigned modifiers) { return 0 != (modifiers & MOD_CONTROL) ; }
inline bool SGLM_Modifiers::IsAlt(    unsigned modifiers) { return 0 != (modifiers & MOD_ALT) ; }
inline bool SGLM_Modifiers::IsSuper(  unsigned modifiers) { return 0 != (modifiers & MOD_SUPER) ; }

inline std::string SGLM_Modifiers::Desc(unsigned modifiers)
{
    std::stringstream ss ; 
    ss << "SGLM_Modifiers::Desc[" 
       << ( IsNone(   modifiers) ? " NONE"    : "" )     
       << ( IsShift(  modifiers) ? " SHIFT"   : "" )     
       << ( IsControl(modifiers) ? " CONTROL" : "" )   
       << ( IsAlt(    modifiers) ? " ALT"     : "" ) 
       << ( IsSuper(  modifiers) ? " SUPER"   : "" )
       <<  "]"
       ;
    std::string str = ss.str();
    return str ; 
}




struct SGLM_Modnav
{
    // NB these enum values must not clash with SGLM_Modifiers
    enum { 
           MOD_W       = 0x1 << 4,
           MOD_A       = 0x1 << 5,
           MOD_S       = 0x1 << 6,
           MOD_D       = 0x1 << 7,
           MOD_Z       = 0x1 << 8,
           MOD_X       = 0x1 << 9,
           MOD_Q       = 0x1 << 10,
           MOD_E       = 0x1 << 11,
           MOD_R       = 0x1 << 12,
           MOD_Y       = 0x1 << 13,
           MOD_ALL     = MOD_W | MOD_A | MOD_S | MOD_D | MOD_Z | MOD_X | MOD_Q | MOD_E | MOD_R | MOD_Y 
          };

    static bool IsNone(unsigned modifiers);
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



inline bool SGLM_Modnav::IsNone(   unsigned modifiers) { return 0 == (modifiers & MOD_ALL  ) ; }
inline bool SGLM_Modnav::IsW(      unsigned modifiers) { return 0 != (modifiers & MOD_W) ; }
inline bool SGLM_Modnav::IsA(      unsigned modifiers) { return 0 != (modifiers & MOD_A) ; }
inline bool SGLM_Modnav::IsS(      unsigned modifiers) { return 0 != (modifiers & MOD_S) ; }
inline bool SGLM_Modnav::IsD(      unsigned modifiers) { return 0 != (modifiers & MOD_D) ; }
inline bool SGLM_Modnav::IsZ(      unsigned modifiers) { return 0 != (modifiers & MOD_Z) ; }
inline bool SGLM_Modnav::IsX(      unsigned modifiers) { return 0 != (modifiers & MOD_X) ; }
inline bool SGLM_Modnav::IsQ(      unsigned modifiers) { return 0 != (modifiers & MOD_Q) ; }
inline bool SGLM_Modnav::IsE(      unsigned modifiers) { return 0 != (modifiers & MOD_E) ; }
inline bool SGLM_Modnav::IsR(      unsigned modifiers) { return 0 != (modifiers & MOD_R) ; }
inline bool SGLM_Modnav::IsY(      unsigned modifiers) { return 0 != (modifiers & MOD_Y) ; }

inline std::string SGLM_Modnav::Desc(unsigned modifiers)
{
    std::stringstream ss ; 
    ss << "SGLM_Modnav::Desc[" 
       << ( IsNone(   modifiers) ? " NONE"    : "" )     
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



