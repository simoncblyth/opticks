#pragma once
/**
sfr.h
======

Unionized glm::tvec4 see examples/UseGLMSimple (too complex, keep it simple). 

**/

#include "NP.hh"
#include "stra.h"

struct sfr
{
    static constexpr const char* NAME = "sfr" ;
    static constexpr const unsigned NUM_4x4 = 3 ; 
    static constexpr const unsigned NUM_VALUES = NUM_4x4*4*4 ; 
    static constexpr const double   EPSILON = 1e-5 ; 
    static constexpr const char* DEFAULT_NAME = "ALL" ;  
    static sfr Make(double extent); 

    glm::tvec4<double>  ce  ;
    glm::tvec4<int64_t> aux0 ; 
    glm::tvec4<int64_t> aux1 ; 
    glm::tvec4<int64_t> aux2 ; 

    glm::tmat4x4<double>  m2w ; 
    glm::tmat4x4<double>  w2m ; 

    const char* name ; // hmm: pointers cause double ownership on default copy issues, use string? 

    sfr(); 

    double* ce_data() ; 
    template<typename T> void set_ce( const T* _ce ); 
    template<typename T> void set_extent( T _w ); 

    void set_name( const char* _name ); 
    const char* get_name() const ; 
    std::string get_key() const ; 

    std::string desc() const ; 
    bool is_identity() const ; 

    NP* serialize() const ; 
    void save(const char* dir, const char* name=NAME) const ; 

    static sfr Import( const NP* a); 
    static sfr Load( const char* dir, const char* name=NAME); 
    static sfr Load_(const char* path ); 

    void load(const char* dir, const char* name=NAME) ; 
    void load_(const char* path ) ; 
    void load(const NP* a) ; 

    double* data() ; 
    const double* cdata() const ; 
    void write( double* dst, unsigned num_values ) const ;
    void read( const double* src, unsigned num_values ) ; 
}; 


inline sfr sfr::Make(double extent)
{
    sfr fr ;
    fr.set_extent(extent); 
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


inline const char* sfr::get_name() const 
{
    return name ; 
}
inline std::string sfr::get_key() const
{
    return name ? sstr::Replace( name, ':', '_' ) : "" ;  
}

inline void sfr::set_name(const char* _n)
{
    name = _n ? strdup(_n) : nullptr ; 
}



inline std::string sfr::desc() const
{
    std::stringstream ss ; 
    ss 
       << "[sfr::desc name [" << ( name ? name : "-" ) << "]\n"
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
    if(name) a->set_meta<std::string>("name",    name ); 
    return a ; 
}

inline void sfr::save(const char* dir, const char* name_ ) const
{
    std::string name = U::form_name( name_ , ".npy" ) ; 
    NP* a = serialize() ; 
    a->save(dir, name.c_str()); 
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
    std::string name = U::form_name( name_ , ".npy" ) ; 
    const NP* a = NP::Load(dir, name.c_str() ); 
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
    if(!_name.empty()) name = strdup(_name.c_str()); 
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

 
