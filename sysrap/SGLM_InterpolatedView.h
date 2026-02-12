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
    float m_df ;

    std::vector<SGLM_View> vv = {} ;


    SGLM_InterpolatedView();
    void setControlledView( SGLM_View* view );

    std::string desc() const ;
    std::string desc_vv() const ;


    void tick();  // primary method, invoking the below

    void setPair(int i, int j);
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
    m_df(0.01)
{
}

inline void SGLM_InterpolatedView::setControlledView( SGLM_View* view )
{
    m_view = view ;
}


inline std::string SGLM_InterpolatedView::desc() const
{
    size_t num_v = vv.size();
    std::stringstream ss ;
    ss << "IV[" << num_v << "] " << ( m_view ? m_view->desc() : "no-controlled-view" ) ;
    std::string str = ss.str() ;
    return str ;
}

inline std::string SGLM_InterpolatedView::desc_vv() const
{
    std::stringstream ss ;
    size_t num_v = vv.size();
    ss << "[SGLM_InterpolatedView::desc_vv num_v " << num_v << "\n" ;
    for(size_t i=0 ; i < num_v ; i++ ) ss << vv[i].desc() << "\n" ;
    ss << "]SGLM_InterpolatedView::desc_vv num_v " << num_v << "\n" ;
    std::string str = ss.str() ;
    return str ;
}




void SGLM_InterpolatedView::tick()
{
    if(m_f + m_df > 1.f )
    {
        m_f = 0.f ;
        nextPair();
    }
    else
    {
        m_f += m_df ;
    }
    updateControlledView();
}

void SGLM_InterpolatedView::nextPair()
{
    int n = vv.size();
    int i = (m_i + 1) % n ;
    int j = (m_j + 1) % n ;
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
    unsigned num = vv.size();
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



