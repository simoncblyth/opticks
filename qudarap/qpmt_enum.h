#pragma once
/**
qpmt_enum.h
=============


**/

enum {
  qpmt_RINDEX,
  qpmt_KINDEX,
  qpmt_QESHAPE,
  qpmt_CETHETA,
  qpmt_CATSPEC,
  qpmt_SPEC,
  qpmt_SPEC_ce,
  qpmt_ART,
  qpmt_COMP,
  qpmt_LL,
  qpmt_ARTE,
  qpmt_ARTE_ce
};


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include "QUDARAP_API_EXPORT.hh"
struct qpmt_enum
{
    static constexpr const char* _qpmt_RINDEX  = "qpmt_RINDEX" ;
    static constexpr const char* _qpmt_KINDEX  = "qpmt_KINDEX" ;
    static constexpr const char* _qpmt_QESHAPE = "qpmt_QESHAPE" ;
    static constexpr const char* _qpmt_CETHETA = "qpmt_CETHETA" ;
    static constexpr const char* _qpmt_CATSPEC = "qpmt_CATSPEC" ;
    static constexpr const char* _qpmt_SPEC    = "qpmt_SPEC" ;
    static constexpr const char* _qpmt_SPEC_ce = "qpmt_SPEC_ce" ;
    static constexpr const char* _qpmt_ART     = "qpmt_ART" ;
    static constexpr const char* _qpmt_COMP    = "qpmt_COMP" ;
    static constexpr const char* _qpmt_LL      = "qpmt_LL" ;
    static constexpr const char* _qpmt_ARTE    = "qpmt_ARTE" ;
    static constexpr const char* _qpmt_ARTE_ce = "qpmt_ARTE_ce" ;

    static const char* Label( int e );
};

inline const char* qpmt_enum::Label(int e)
{
    const char* s = nullptr ;
    switch(e)
    {
        case qpmt_RINDEX:  s = _qpmt_RINDEX  ; break ;
        case qpmt_KINDEX:  s = _qpmt_KINDEX  ; break ;
        case qpmt_QESHAPE: s = _qpmt_QESHAPE ; break ;
        case qpmt_CETHETA: s = _qpmt_CETHETA ; break ;
        case qpmt_CATSPEC: s = _qpmt_CATSPEC ; break ;
        case qpmt_SPEC:    s = _qpmt_SPEC    ; break ;
        case qpmt_SPEC_ce: s = _qpmt_SPEC_ce ; break ;
        case qpmt_ART:     s = _qpmt_ART     ; break ;
        case qpmt_COMP:    s = _qpmt_COMP    ; break ;
        case qpmt_LL:      s = _qpmt_LL      ; break ;
        case qpmt_ARTE:    s = _qpmt_ARTE    ; break ;
        case qpmt_ARTE_ce: s = _qpmt_ARTE_ce ; break ;
    }
    return s ;
}
#endif


