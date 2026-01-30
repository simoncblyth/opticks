#pragma once
/**
sfr.h
======

Unionized glm::tvec4 see examples/UseGLMSimple (too complex, keep it simple).

**/

#include "NP.hh"
#include "sstr.h"
#include "stra.h"



struct sfr
{
    static constexpr const char* NAME = "sfr" ;
    static constexpr const unsigned NUM_4x4 = 3 ;
    static constexpr const unsigned NUM_VALUES = NUM_4x4*4*4 ;
    static constexpr const double   EPSILON = 1e-5 ;
    static constexpr const char* DEFAULT_NAME = "ALL" ;

    template<typename T>
    static sfr MakeFromCE(const T* ce);


    template<typename T>
    static sfr MakeFromExtent(const char* _extent);
    template<typename T>
    static sfr MakeFromExtent(T extent);

    template<typename T>
    static sfr MakeFromAxis(const char* tpde, char delim=',');
    template<typename T>
    static sfr MakeFromAxis(T theta_deg, T phi_deg, T ax_dist_mm, T extent_mm);


    glm::tvec4<double>  ce  ;    //  4*8 = 32       0
    glm::tvec4<int64_t> aux0 ;   //  4*8 = 32      32
    glm::tvec4<int64_t> aux1 ;   //  4*8 = 32      64
    glm::tvec4<int64_t> aux2 ;   //  4*8 = 32      96

    glm::tmat4x4<double>  m2w ;  // 4*4*8 = 128   128
    glm::tmat4x4<double>  w2m ;  // 4*4*8 = 128   256

    std::string name ;           //               384

    // bytewise comparison of sfr instances fails
    // for 4 bytes at offset 384 corresponding to the std::string name reference

    sfr();

    double* ce_data() ;
    template<typename T> void set_ce( const T* _ce );
    template<typename T> void set_extent( T _w );
    template<typename T> void set_m2w( const T* _v16, size_t nv=16 );

    void set_idx(int idx) ;
    int  get_idx() const ;

    void set_name( const char* _name );
    const std::string& get_name() const ;
    std::string get_key() const ;

    std::string desc_ce() const ;
    std::string desc() const ;
    bool is_identity() const ;

    NP* serialize() const ;
    void save(const char* dir, const char* stem=NAME) const ;

    static sfr Import( const NP* a);
    static sfr Load( const char* dir, const char* stem=NAME);
    static sfr Load_(const char* path );

    void load(const char* dir, const char* stem=NAME) ;
    void load_(const char* path ) ;
    void load(const NP* a) ;

    double* data() ;
    const double* cdata() const ;
    void write( double* dst, unsigned num_values ) const ;
    void read( const double* src, unsigned num_values ) ;
};


template<typename T>
inline sfr sfr::MakeFromCE(const T* _ce )
{
    sfr fr ;
    fr.set_ce(_ce);
    return fr ;
}





template<typename T>
inline sfr sfr::MakeFromExtent(const char* _ext)
{
    T _extent = sstr::To<T>( _ext ) ;
    return MakeFromExtent<T>(_extent);
}

template<typename T>
inline sfr sfr::MakeFromExtent(T extent)
{
    sfr fr ;
    fr.set_extent(extent);
    return fr ;
}


/**
sfr::MakeFromAxis
------------------

::

    MOI=AXIS:56,-54,-21271,5000 cxr_min.sh

    ELV=^s_EMF EYE=0,0,-2 UP=0,1,0 MOI=AXIS:56,-54,-21271,3843 cxr_min.sh


    // EMF_ZC_32 / EMF_RC_32:
    //   32 tilted rings (holders) defined in cylindrical coordinates around
    //   the tilted axis a (θ = 56°, φ = −54°).
    //   - Zc : coordinate along axis a, measured from the world origin (m).
    //   - Rc : radial distance from axis a to the “neutral layer” of the ring (m).
    const double WaterPoolConstructionEMF::EMF_ZC_32[WaterPoolConstructionEMF::kHolderRingCount] = {
        21.271,  20.602,  19.554,  18.555,
        17.077,  15.600,  14.122,  12.644,
        11.166,   9.685,   8.207,   6.729,
         5.251,   3.773,   2.295,   0.817,
        -0.821,  -2.299,  -3.777,  -5.255,
        -6.733,  -8.210,  -9.688, -11.166,
       -12.644, -14.122, -15.600, -17.077,
       -18.555, -19.554, -20.602, -21.271
    };

    const double WaterPoolConstructionEMF::EMF_RC_32[WaterPoolConstructionEMF::kHolderRingCount] = {
        3.843,   6.509,   9.185,  11.053,
       13.210,  14.924,  16.325,  17.495,
       18.465,  19.282,  19.951,  20.502,
       20.930,  21.240,  21.450,  21.559,
       21.559,  21.450,  21.240,  20.930,
       20.502,  19.951,  19.282,  18.465,
       17.495,  16.325,  14.924,  13.210,
       11.053,   9.185,   6.509,   3.843
    };



**/


template<typename T>
inline sfr sfr::MakeFromAxis(const char* tpde, char delim)
{
    std::vector<T> elem ;
    sstr::split<T>( elem, tpde, delim );
    int num_elem = elem.size();

    T theta_deg = num_elem > 0 ? elem[0] : 0. ;
    T phi_deg   = num_elem > 1 ? elem[1] : 0. ;
    T dist_mm   = num_elem > 2 ? elem[2] : 0. ;
    T extent_mm = num_elem > 3 ? elem[3] : 1000. ;

    std::cout
        << "sfr::MakeFromAxis"
        << " tpde [" << ( tpde ? tpde : "-" ) << "]"
        << " num_elem " << num_elem
        << " elem " << sstr::desc<T>(elem)
        << "\n"
        ;

    return MakeFromAxis<T>( theta_deg, phi_deg, dist_mm, extent_mm );
}



template<typename T>
inline sfr sfr::MakeFromAxis(T theta_deg, T phi_deg, T ax_dist_mm, T extent_mm)
{
    std::cout
        << "sfr::MakeFromAxis"
        << " theta_deg " << theta_deg
        << " phi_deg " << phi_deg
        << " ax_dist_mm " << ax_dist_mm
        << " extent_mm " << extent_mm
        << "\n"
        ;

    T theta = theta_deg * glm::pi<T>() / 180.;
    T phi = phi_deg * glm::pi<T>() / 180.;

    T st = glm::sin(theta);
    T ct = glm::cos(theta);
    T sp = glm::sin(phi);
    T cp = glm::cos(phi);

    glm::tvec3<T> ax = glm::vec3(st * cp, st * sp, ct);
    glm::tvec3<T> translation = ax * ax_dist_mm ;

    glm::tvec3<T> world_z = glm::tvec3<T>(0.0f, 0.0f, 1.0f);
    glm::tvec3<T> up = world_z - glm::dot(world_z, ax) * ax;
    up = glm::normalize(up);

    glm::tmat4x4<T> model2world = stra<T>::Model2World(ax, up, translation );

    sfr fr ;
    fr.set_m2w( glm::value_ptr(model2world) );
    fr.set_extent(extent_mm);

    return fr ;
}




inline sfr::sfr()
    :
    ce(0.,0.,0.,100.),
    aux0(0),
    aux1(0),
    aux2(0),
    m2w(1.),
    w2m(1.),
    name(DEFAULT_NAME)
{
}

inline double* sfr::ce_data()
{
    return glm::value_ptr(ce);
}

template<typename T>
inline void sfr::set_ce( const T* _ce )
{
    ce.x = _ce[0];
    ce.y = _ce[1];
    ce.z = _ce[2];
    ce.w = _ce[3];
}

template<typename T>
inline void sfr::set_extent( T _w )
{
    ce.w = _w ;
}

template<typename T>
inline void sfr::set_m2w( const T* vv, size_t nv )
{
    assert( nv == 16 );
    double* _m2w = glm::value_ptr(m2w) ;
    for(size_t i=0 ; i < nv ; i++ ) _m2w[i] = T(vv[i]);
    w2m = glm::inverse(m2w);
}






inline void sfr::set_idx(int idx) { aux2.w = idx ; }
inline int  sfr::get_idx() const  { return aux2.w ; }


inline const std::string& sfr::get_name() const
{
    return name ;
}
inline std::string sfr::get_key() const
{
    return name.empty() ? "" : sstr::Replace( name.c_str(), ':', '_' ) ;
}

inline void sfr::set_name(const char* _n)
{
    if(_n) name = _n ;
}


inline std::string sfr::desc_ce() const
{
    std::stringstream ss ;
    ss << "sfr::desc_ce " << stra<double>::Desc(ce) ;
    std::string str = ss.str();
    return str ;
}

inline std::string sfr::desc() const
{
    std::stringstream ss ;
    ss
       << "[sfr::desc name [" << name << "]\n"
       << "ce\n"
       << stra<double>::Desc(ce)
       << "\n"
       << "aux0\n"
       << stra<int64_t>::Desc(aux0)
       << "\n"
       << "aux1\n"
       << stra<int64_t>::Desc(aux1)
       << "\n"
       << "aux2\n"
       << stra<int64_t>::Desc(aux2)
       << "\n"
       << "m2w\n"
       << stra<double>::Desc(m2w)
       << "\n"
       << "w2m\n"
       << stra<double>::Desc(w2m)
       << "\n"
       << "is_identity " << ( is_identity() ? "YES" : "NO " ) << "\n"
       << "]sfr::desc\n"
       ;

    std::string str = ss.str();
    return str ;
}

inline bool sfr::is_identity() const
{
    bool m2w_identity = stra<double>::IsIdentity(m2w, EPSILON);
    bool w2m_identity = stra<double>::IsIdentity(w2m, EPSILON);
    return m2w_identity && w2m_identity ;
}





inline NP* sfr::serialize() const
{
    NP* a = NP::Make<double>(NUM_4x4, 4, 4) ;
    write( a->values<double>(), NUM_4x4*4*4 ) ;
    a->set_meta<std::string>("creator", "sfr::serialize");
    if(!name.empty()) a->set_meta<std::string>("name",    name );
    return a ;
}

inline void sfr::save(const char* dir, const char* stem_ ) const
{
    std::string aname = U::form_name( stem_ , ".npy" ) ;
    NP* a = serialize() ;
    a->save(dir, aname.c_str());
}







inline sfr sfr::Import( const NP* a) // static
{
    sfr fr ;
    fr.load(a);
    return fr ;
}

inline sfr sfr::Load(const char* dir, const char* name) // static
{
    sfr fr ;
    fr.load(dir, name);
    return fr ;
}
inline sfr sfr::Load_(const char* path) // static
{
    sfr fr ;
    fr.load_(path);
    return fr ;
}


inline void sfr::load(const char* dir, const char* name_ )
{
    std::string aname = U::form_name( name_ , ".npy" ) ;
    const NP* a = NP::Load(dir, aname.c_str() );
    load(a);
}
inline void sfr::load_(const char* path_)
{
    const NP* a = NP::Load(path_);
    if(!a) std::cerr
       << "sfr::load_ ERROR : non-existing"
       << " path_ " << path_
       << std::endl
       ;
    assert(a);
    load(a);
}
inline void sfr::load(const NP* a)
{
    read( a->cvalues<double>() , NUM_VALUES );
    std::string _name = a->get_meta<std::string>("name", "");
    if(!_name.empty()) name = _name ;
}

inline const double* sfr::cdata() const
{
    return (const double*)&ce.x ;
}
inline double* sfr::data()
{
    return (double*)&ce.x ;
}
inline void sfr::write( double* dst, unsigned num_values ) const
{
    assert( num_values == NUM_VALUES );
    char* dst_bytes = (char*)dst ;
    char* src_bytes = (char*)cdata();
    unsigned num_bytes = sizeof(double)*num_values ;
    memcpy( dst_bytes, src_bytes, num_bytes );
}

inline void sfr::read( const double* src, unsigned num_values )
{
    assert( num_values == NUM_VALUES );
    char* src_bytes = (char*)src ;
    char* dst_bytes = (char*)data();
    unsigned num_bytes = sizeof(double)*num_values ;
    memcpy( dst_bytes, src_bytes, num_bytes );
}

inline std::ostream& operator<<(std::ostream& os, const sfr& fr)
{
    os << fr.desc() ;
    return os;
}


