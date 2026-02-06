#pragma once
/**
sfr.h
======

The frame is needed in almost all situations:

* simulate : input photon transformation
* rendering : prepare raytrace rendering param
* simtrace C++ : genstep preparation
* simtrace python : metadata from the persisted frame

Where to keep the frame ?
----------------------------

* for simulation the obvious location is SEvt
* for simtrace it needs to be in SEvt for genstep preparation
* for rendering the obvious location is SGLM

Currently reluctant to depend on SEvt for rendering, because it
brings complexity without much utility. So live with two locations
for the frame.


sfr Layout
-----------

::

     +------------+-------------+-------------+----------------+
     |   ce.x     |    ce.y     |    ce.z     |   ce.w         |
     |            |             |             |                |
     +------------+-------------+-------------+----------------+
     |   aux0.x   |   aux0.y    |   aux0.z    |  aux0.w        |
     |            |             |             |                |
     +------------+-------------+-------------+----------------+
     |   aux1.x   |   aux1.y    |   aux1.z    |  aux1.w        |
     |   hss      |   gas       |   sensorid  |  sensoridx     |
     +------------+-------------+-------------+----------------+
     |   aux2.x   |   aux2.y    |   aux2.z    |  aux2.w        |
     |   inst     |   nidx      |   prim      |  idx           |
     +------------+-------------+-------------+----------------+

     +------------+-------------+-------------+----------------+
     |  m2w                                                    |
     |                                                         |
     |                                                         |
     |                                                         |
     +------------+-------------+-------------+----------------+

     +------------+-------------+-------------+----------------+
     |  w2m                                                    |
     |                                                         |
     |                                                         |
     |                                                         |
     +------------+-------------+-------------+----------------+

     +---------------------------------------------------------+
     |   bbmn.x   |   bbmn.y    |   bbmn.z    |  bbmx.x        |
     |            |             |             |                |
     +------------+-------------+-------------+----------------+
     |   bbmx.y   |   bbmx.z    |   padd.x    |  padd.y        |
     |            |             |             |                |
     +------------+-------------+-------------+----------------+
     |  ext0.x    |   ext0.y    |   ext0.z    |  ext0.w        |
     |   eps      |   gridscale |             |                |
     +------------+-------------+-------------+----------------+
     |  ext1.x    |   ext1.y    |   ext1.z    |  ext1.w        |
     |            |             |             |                |
     +------------+-------------+-------------+----------------+

**/

#include "NP.hh"
#include "sstr.h"
#include "stra.h"
#include "stran.h"
#include "sphoton.h"

struct sfr
{
    static constexpr const char* NAME = "sfr" ;
    static constexpr const unsigned NUM_4x4 = 4 ;
    static constexpr const unsigned NUM_VALUES = NUM_4x4*4*4 ;  // 64
    static constexpr const double   EPSILON = 1e-5 ;
    static constexpr const char* DEFAULT_NAME = "ALL" ;


    template<typename T>
    static sfr MakeFromCE(const char* ce, char delim=',');
    template<typename T>
    static sfr MakeFromCE(const T* ce);

    template<typename T>
    static sfr MakeFromExtent(const char* _extent);
    template<typename T>
    static sfr MakeFromExtent(T extent);


    template<typename T>
    static sfr MakeFromTranslateExtent(const char* s_te, char delim);
    template<typename T>
    static sfr MakeFromTranslateExtent(const T* _te);
    template<typename T>
    static sfr MakeFromTranslateExtent(T tx, T ty, T tz, T extent );



    template<typename T>
    static sfr MakeFromAxis(const char* tpde, char delim=',');
    template<typename T>
    static sfr MakeFromAxis(T theta_deg, T phi_deg, T ax_dist_mm, T extent_mm, T delta_ax_distance_mm );

                                 //  nv   nv_offset
    glm::tvec4<double>  ce  ;    //  4       0
    glm::tvec4<int64_t> aux0 ;   //  4       4
    glm::tvec4<int64_t> aux1 ;   //  4       8
    glm::tvec4<int64_t> aux2 ;   //  4      12       1st 4x4

    glm::tmat4x4<double>  m2w ;  //  16     16       2nd 4x4
    glm::tmat4x4<double>  w2m ;  //  16     32       3rd 4x4

    glm::tvec3<double>  bbmn ;   //   3     48
    glm::tvec3<double>  bbmx ;   //   3     51
    glm::tvec2<double>  padd ;   //   2     54
    glm::tvec4<double>  ext0 ;   //   4     56
    glm::tvec4<double>  ext1 ;   //   4     60       4th 4x4
                                 //   -     64
    std::string name ;

    // bytewise comparison of sfr instances fails
    // for 4 bytes at offset corresponding to the std::string name reference

    sfr();

    void set_propagate_epsilon(double eps);
    void set_gridscale(double gsc);

    double get_propagate_epsilon() const ;
    double get_gridscale() const ;

    void set_hostside_simtrace();
    bool  is_hostside_simtrace() const ;

    void set_gasix(   int gix) ;
    void set_sensorid(int sid) ;
    void set_sensorix(int six) ;

    int get_gasix() const ;
    int get_sensorid() const ;
    int get_sensorix() const ;

    void  set_identity( int inst, int gasix, int sensorid, int sensorix );

    void set_inst(int idx) ;
    void set_nidx(int nidx) ;
    void set_prim(int prim) ;
    void set_idx(int idx) ;

    int  get_inst() const ;
    int  get_nidx() const ;
    int  get_prim() const ;
    int  get_idx() const ;



    bool is_zero() const ;


    double* ce_data() ;
    template<typename T> void set_ce( const T* _ce );
    template<typename T> void set_extent( T _w );
    template<typename T> void set_m2w( const T* _v16, size_t nv=16 );
    template<typename T> void set_bb(  const T* _bb6 );

    const glm::tmat4x4<double>& get_transform(bool inverse) const ;

    void transform_w2m( sphoton& p, bool normalize=true ) const ;
    void transform_m2w( sphoton& p, bool normalize=true ) const ;
    void transform(     sphoton& p, bool normalize, bool inverse ) const ;

    Tran<double>* getTransform() const;
    NP* transform_photon_m2w( const NP* ph, bool normalize ) const ;
    NP* transform_photon_w2m( const NP* ph, bool normalize ) const ;
    NP* transform_photon(     const NP* ph, bool normalize, bool inverse ) const ;



    void set_name( const char* _name );
    const std::string& get_name() const ;
    const char* get_id() const ;

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
inline sfr sfr::MakeFromCE(const char* s_ce, char delim)
{
    std::vector<T> elem ;
    sstr::split<T>( elem, s_ce, delim );
    int num_elem = elem.size();

    std::array<T,4> _ce ;

    _ce[0] = num_elem > 0 ? elem[0] : 0. ;
    _ce[1] = num_elem > 1 ? elem[1] : 0. ;
    _ce[2] = num_elem > 2 ? elem[2] : 0. ;
    _ce[3] = num_elem > 3 ? elem[3] : 1000. ;

    return MakeFromCE<T>(_ce.data());
}

template<typename T>
inline sfr sfr::MakeFromCE(const T* _ce )
{
    sfr fr ;
    fr.set_ce(_ce);
    fr.set_name("MakeFromCE");
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
    fr.set_name("MakeFromExtent");
    return fr ;
}








template<typename T>
inline sfr sfr::MakeFromTranslateExtent(const char* s_te, char delim)
{
    std::vector<T> elem ;
    sstr::split<T>( elem, s_te, delim );
    int num_elem = elem.size();

    std::array<T,4> _te ;

    _te[0] = num_elem > 0 ? elem[0] : 0. ;
    _te[1] = num_elem > 1 ? elem[1] : 0. ;
    _te[2] = num_elem > 2 ? elem[2] : 0. ;
    _te[3] = num_elem > 3 ? elem[3] : 1000. ;

    return MakeFromTranslateExtent<T>(_te.data());
}

template<typename T>
inline sfr sfr::MakeFromTranslateExtent(const T* _te )
{
    return MakeFromTranslateExtent(_te[0], _te[1], _te[2], _te[3]);
}

template<typename T>
inline sfr sfr::MakeFromTranslateExtent(T tx, T ty, T tz, T extent )
{
    T sc = 1. ;
    glm::tmat4x4<T> model2world = stra<T>::Translate(tx, ty, tz, sc );
    sfr fr ;
    fr.set_m2w( glm::value_ptr(model2world) );
    fr.set_extent(extent);
    fr.set_name("MakeFromTranslateExtent");
    return fr ;
}

















/**
sfr::MakeFromAxis
------------------

::

    MOI=AXIS:56,-54,-21271,5000 cxr_min.sh

    ELV=^s_EMF EYE=0,0,-2 UP=0,1,0 MOI=AXIS:56,-54,-21271,3843 cxr_min.sh

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
    T delta_dist_mm   = num_elem > 4 ? elem[4] : 0. ;

    std::cout
        << "sfr::MakeFromAxis"
        << " tpde [" << ( tpde ? tpde : "-" ) << "]"
        << " num_elem " << num_elem
        << " elem " << sstr::desc<T>(elem)
        << "\n"
        ;

    return MakeFromAxis<T>( theta_deg, phi_deg, dist_mm, extent_mm, delta_dist_mm );
}



template<typename T>
inline sfr sfr::MakeFromAxis(T theta_deg, T phi_deg, T ax_dist_mm, T extent_mm, T delta_ax_dist_mm )
{
    std::cout
        << "sfr::MakeFromAxis"
        << " theta_deg " << theta_deg
        << " phi_deg " << phi_deg
        << " ax_dist_mm " << ax_dist_mm
        << " extent_mm " << extent_mm
        << " delta_ax_dist_mm " << delta_ax_dist_mm
        << "\n"
        ;

    T theta = theta_deg * glm::pi<T>() / 180.;
    T phi = phi_deg * glm::pi<T>() / 180.;

    T st = glm::sin(theta);
    T ct = glm::cos(theta);
    T sp = glm::sin(phi);
    T cp = glm::cos(phi);

    glm::tvec3<T> ax = glm::vec3(st * cp, st * sp, ct);
    glm::tvec3<T> translation = ax * ( ax_dist_mm + delta_ax_dist_mm ) ;

    glm::tvec3<T> world_z = glm::tvec3<T>(0.0f, 0.0f, 1.0f);
    glm::tvec3<T> up = world_z - glm::dot(world_z, ax) * ax;
    up = glm::normalize(up);

    glm::tmat4x4<T> model2world = stra<T>::Model2World(ax, up, translation );

    sfr fr ;
    fr.set_m2w( glm::value_ptr(model2world) );
    fr.set_extent(extent_mm);
    fr.set_name("MakeFromAxis");

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
    bbmn(0.,0.,0.),
    bbmx(0.,0.,0.),
    padd(0.,0.),
    ext0(0.,0.,0.,0.),
    ext1(0.,0.,0.,0.),
    name(DEFAULT_NAME)
{
}



inline void sfr::set_propagate_epsilon(double eps){     ext0.x = eps ; }
inline void sfr::set_gridscale(double gsc){             ext0.y = gsc ; }
inline double sfr::get_propagate_epsilon() const   {  return ext0.x ; }
inline double sfr::get_gridscale() const   {         return ext0.y ; }

inline void   sfr::set_hostside_simtrace(){     aux1.x = 1 ; }   // hss
inline void   sfr::set_gasix( int gasix ){      aux1.y = gasix ; }
inline void   sfr::set_sensorid(  int senid ){  aux1.z = senid ; }
inline void   sfr::set_sensorix(  int senix ){  aux1.w = senix ; }

inline bool   sfr::is_hostside_simtrace() const { return aux1.x == 1 ; }  // hss
inline int    sfr::get_gasix() const {            return aux1.y ; }
inline int    sfr::get_sensorid() const {         return aux1.z ; }
inline int    sfr::get_sensorix() const {         return aux1.w ; }

inline void   sfr::set_identity( int inst, int gasix, int sensorid, int sensorix )
{
    set_inst(inst);
    set_gasix(gasix);
    set_sensorid( sensorid );
    set_sensorix( sensorix );
}

inline void sfr::set_inst(int ii){     aux2.x = ii ;   }
inline void sfr::set_nidx(int nidx){   aux2.y = nidx ; }
inline void sfr::set_prim(int prim){   aux2.z = prim ; }
inline void sfr::set_idx(int idx) {    aux2.w = idx ; }

inline int  sfr::get_inst() const  { return aux2.x ; }
inline int  sfr::get_nidx() const  { return aux2.y ; }
inline int  sfr::get_prim() const  { return aux2.z ; }
inline int  sfr::get_idx() const  {  return aux2.w ; }










inline bool sfr::is_zero() const
{
    return ce.x == 0. && ce.y == 0. && ce.z == 0. && ce.w == 0. ;
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


template<typename T>
inline void sfr::set_bb( const T* bb )
{
    bbmn.x = bb[0] ;
    bbmn.y = bb[1] ;
    bbmn.z = bb[2] ;
    bbmx.x = bb[3] ;
    bbmx.y = bb[4] ;
    bbmx.z = bb[5] ;
}








inline const glm::tmat4x4<double>& sfr::get_transform(bool inverse) const
{
    return inverse ? w2m : m2w ;
}
inline void sfr::transform_w2m( sphoton& p, bool normalize ) const
{
    transform( p, normalize, true );
}
inline void sfr::transform_m2w( sphoton& p, bool normalize ) const
{
    transform( p, normalize, false );
}
inline void sfr::transform( sphoton& p, bool normalize, bool inverse ) const
{
    const glm::tmat4x4<double>& tr = get_transform(inverse);
    p.transform( tr, normalize );
}








/**
sfr::getTransform
-------------------

t:m2w
v:w2m

**/

inline Tran<double>* sfr::getTransform() const
{
    Tran<double>* geotran = new Tran<double>( m2w, w2m );   // ORDER ?
    return geotran ;
}

inline NP* sfr::transform_photon_m2w( const NP* ph, bool normalize ) const
{
    bool inverse = false ; // false:m2w true:w2m
    return transform_photon(ph, normalize, inverse);
}
inline NP* sfr::transform_photon_w2m( const NP* ph, bool normalize ) const
{
    bool inverse = true ; // false:m2w true:w2m
    return transform_photon(ph, normalize, inverse);
}


/**
sfr::transform_photon
----------------------

Canonical call from::

     SEvt::transformInputPhoton
     SEvt::setFr


When normalize is true the mom and pol are normalized after the transformation.
Note that the transformed photon array is always in double precision.
That will be narrowed down to float prior to upload by QEvt::setInputPhoton

**/

inline NP* sfr::transform_photon( const NP* ph, bool normalize, bool inverse ) const
{
    if( ph == nullptr ) return nullptr ;
    Tran<double>* tr = getTransform();
    assert(tr);
    NP* pht = Tran<double>::PhotonTransform(ph, normalize, tr, inverse );
    assert( pht->ebyte == 8 );
    return pht ;
}



inline const std::string& sfr::get_name() const
{
    return name ;
}
inline const char* sfr::get_id() const
{
    return name.c_str() ;
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
       << "bbmn\n"
       << stra<double>::Desc(bbmn)
       << "\n"
       << "bbmx\n"
       << stra<double>::Desc(bbmx)
       << "\n"
       << "padd\n"
       << stra<double>::Desc(padd)
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


