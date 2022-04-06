#pragma once
/**
sflow.h
==========

Flow control enum and CPU only presentation

**/

enum { 
   UNDEFINED, 
   BREAK, 
   CONTINUE, 
   BOUNDARY,
   PASS, 
   START, 
   RETURN,
   LAST
   }; 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
struct sflow
{
    static const char* desc( unsigned ctrl ); 
    static constexpr const char* UNDEFINED_ = "UNDEFINED";
    static constexpr const char* BREAK_     = "BREAK";
    static constexpr const char* CONTINUE_  = "CONTINUE" ;
    static constexpr const char* BOUNDARY_  = "BOUNDARY" ;
    static constexpr const char* PASS_      = "PASS" ;
    static constexpr const char* START_     = "START" ;
    static constexpr const char* RETURN_    = "RETURN" ;
    static constexpr const char* LAST_      = "LAST" ;
};
inline const char* sflow::desc(unsigned ctrl )
{
    const char* d = nullptr ; 
    switch(ctrl)
    {
        case UNDEFINED: d = UNDEFINED_ ; break ; 
        case BREAK    : d = BREAK_     ; break ; 
        case CONTINUE : d = CONTINUE_  ; break ; 
        case BOUNDARY : d = BOUNDARY_  ; break ; 
        case PASS:      d = PASS_      ; break ; 
        case START:     d = START_     ; break ; 
        case RETURN:    d = RETURN_    ; break ; 
        case LAST:      d = LAST_      ; break ; 
    }
    return d ; 
}
#endif
 
