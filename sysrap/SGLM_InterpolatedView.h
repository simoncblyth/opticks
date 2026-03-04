#pragma once

#include <vector>
#include "SGLM_View.h"
#include <glm/gtx/spline.hpp>


struct SGLM_InterpolatedView
{
    static SGLM_InterpolatedView* Load(const char* path);
    void load( const char* path );
    void import_( const NP* views );



    SGLM_View* m_view ;
    int   m_i ;
    int   m_j ;
    float m_f ;
    float m_df0 ;
    float m_df ;
    int   m_ifly_value ;

    std::vector<SGLM_View> vv = {} ;


    SGLM_InterpolatedView();
    void setControlledView( SGLM_View* view );

    std::string brief() const ;
    std::string desc() const ;
    std::string detail() const ;


    void tick(int ifly_value, bool ifly_flip);  // primary method, invoking the below

    void setPair(int i, int j);
    static int Modulus_Positive(int a, int b);
    void nextPair();
    void updateControlledView();


};


inline SGLM_InterpolatedView* SGLM_InterpolatedView::Load(const char* path)
{
    SGLM_InterpolatedView* iv = new SGLM_InterpolatedView ;
    iv->load(path);
    return iv ;
}

inline void SGLM_InterpolatedView::load(const char* path)
{
    const NP* views = NP::Load(path);
    import_(views);
}

inline void SGLM_InterpolatedView::import_(const NP* _views)
{
    unsigned num_view = _views->shape[0] ;
    SGLM_View* views = (SGLM_View*)_views->bytes();
    for(unsigned i=0 ; i < num_view ; i++)
    {
        const SGLM_View& v = views[i];
        vv.push_back(v);
    }
}






inline SGLM_InterpolatedView::SGLM_InterpolatedView()
    :
    m_view(nullptr),
    m_i(0),
    m_j(1),
    m_f(0.f),
    m_df0(0.01),
    m_df(0.01),
    m_ifly_value(0)
{
}

inline void SGLM_InterpolatedView::setControlledView( SGLM_View* view )
{
    m_view = view ;
}

inline std::string SGLM_InterpolatedView::brief() const
{
    size_t num_v = vv.size();
    std::stringstream ss ;
    ss << "IV[" << num_v << "] " << ( m_view ? m_view->desc() : "no-controlled-view" ) ;
    std::string str = ss.str() ;
    return str ;
}

inline std::string SGLM_InterpolatedView::desc() const
{
    size_t num_v = vv.size();
    std::stringstream ss ;
    ss << "IV[" << num_v << "] " << ( m_view ? m_view->desc() : "no-controlled-view" ) ;
    std::string str = ss.str() ;
    return str ;
}

inline std::string SGLM_InterpolatedView::detail() const
{
    std::stringstream ss ;
    size_t num_v = vv.size();
    ss << "[SGLM_InterpolatedView::desc_vv num_v " << num_v << "\n" ;
    for(size_t i=0 ; i < num_v ; i++ ) ss << vv[i].desc() << "\n" ;
    ss << "]SGLM_InterpolatedView::desc_vv num_v " << num_v << "\n" ;
    std::string str = ss.str() ;
    return str ;
}


/**
SGLM_InterpolatedView::tick
----------------------------

TODO: direction+speed control

**/

void SGLM_InterpolatedView::tick(int ifly_value, bool ifly_flip)
{
    m_ifly_value = ifly_value ;
    assert( m_ifly_value != 0 );
    m_df = ( ifly_flip ? -1.f : 1.f )*m_df0*float(m_ifly_value) ;

    if(m_f + m_df > 1.f )
    {
        m_f = 0.f ;
        nextPair();
    }
    else if(m_f + m_df < 0.f )
    {
        m_f = 1.f ;
        nextPair();
    }
    else
    {
        m_f += m_df ;
    }
    updateControlledView();
}


/**
SGLM_InterpolatedView::Modulus_Positive
----------------------------------------

Handle -ve a like python does::

    Modulus_Positive( -2, 10 ) == 8

**/

int SGLM_InterpolatedView::Modulus_Positive(int a, int b)
{
    int r = a % b;
    return r >= 0 ? r : r + b;
}

void SGLM_InterpolatedView::nextPair()
{
    int delta = m_df > 0.f ? 1 : -1 ;
    int n = vv.size();
    int i = Modulus_Positive( m_i + delta, n );
    int j = Modulus_Positive( m_j + delta, n );
    setPair(i,j);
}

void SGLM_InterpolatedView::setPair(int i, int j)
{
    m_i = i ;
    m_j = j ;
}





void SGLM_InterpolatedView::updateControlledView()
{
    assert(m_view);
    int num = vv.size();
    if(num < 2) return;

    unsigned h = (m_i > 0) ? m_i - 1 : m_i;
    unsigned i = m_i ;
    unsigned j = m_j ;
    unsigned k = (m_j + 1 < num) ? m_j + 1 : m_j;

    glm::vec3 Eh = glm::vec3(vv[h].EYE) ;
    glm::vec3 Ei = glm::vec3(vv[i].EYE) ;
    glm::vec3 Ej = glm::vec3(vv[j].EYE) ;
    glm::vec3 Ek = glm::vec3(vv[k].EYE) ;
    glm::vec3 E  = glm::catmullRom(Eh, Ei, Ej, Ek, m_f);

    glm::quat Qi = vv[i].getQuat() ;
    glm::quat Qj = vv[j].getQuat() ;
    glm::quat Q = glm::slerp(Qi, Qj, m_f );

    glm::mat4 QRot = glm::mat4_cast(Q);
    // In camera space, look is -Z, up is +Y. Rotate these into Model Space.
    glm::vec3 _LOOKDIR = -glm::vec3(QRot[0][2], QRot[1][2], QRot[2][2]);
    glm::vec3 _UP      =  glm::vec3(QRot[0][1], QRot[1][1], QRot[2][1]);
    glm::vec3 _LOOK    = E + _LOOKDIR ;

    m_view->EYE  = glm::vec4(E,     1.f);
    m_view->LOOK = glm::vec4(_LOOK, 1.f);
    m_view->UP   = glm::vec4(_UP,   0.f);
}



