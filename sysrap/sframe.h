#pragma once

/**
sframe.h
===========

TODO: once bring in U4Tree.h/stree.h translation up this to double precision 


Provided by CSGFoundry::getFrame methods 

* for MOI lookups CSGTarget::getFrameComponents sets the transforms



Persisted into (4,4,4) array.
Any extension should be in quad4 blocks 
for persisting, alignment and numpy convenience

Note that some variables like *frs* are
persisted in metadata, not in the array. 

Currently *frs* is usually the same as *moi* from MOI envvar
but are using *frs* to indicate intension for generalization 
to frame specification using global instance index rather than MOI
which uses the gas specific instance index. 

TODO: should be using Tran<double> for transforming , might as well 
      do the lot in double : double4,dquad,dqat4 


**/

#include <cassert>
#include <vector>
#include <cstring>
#include <string>
#include <cstdlib>

#include "scuda.h"
#include "squad.h"

#ifdef WITH_SCUDA_DOUBLE
#include "squad_double.h"
#include "scuda_double.h"
#endif

#include "sqat4.h"
#include "stran.h"
#include "spath.h"
#include "sphoton.h"

#include "sfr.h"

#include "NP.hh"


struct sframe
{
    static constexpr const bool VERBOSE = false ; 
    static constexpr const char* NAME = "sframe" ;  // formerly with .npy, now splitting ext for easier stem changes
    static constexpr const char* DEFAULT_FRS = "-1" ; 
    static constexpr const char* DEFAULT_NAME = "ALL" ;  

    static constexpr const unsigned NUM_4x4 = 4 ; 
    static constexpr const unsigned NUM_VALUES = NUM_4x4*4*4 ; 
    static constexpr const float EPSILON = 1e-5 ; 

    float4 ce = {} ;   // f4:center-extent           4x4:0
    quad   q1 = {} ;   // i4:cegs
    quad   q2 = {} ;   // i4:cegs.., f4:gridscale
    quad   q3 = {} ;   // i4:midx,mord,gord,inst 
   
    qat4   m2w = {} ;  // f4:inst transform          4x4:1
    qat4   w2m = {} ;  // f4:iinst transform         4x4:2

    quad4  aux = {} ;  // i4:ins,gas,sen_id,sen_idx  4x4:3


    // CAUTION : ABOVE HEAD PERSISTED BY MEMCPY INTO ARRAY,  BELOW TAIL ADDED AS METADATA

    // on the edge, the above are memcpy in/out by load/save
    const char* frs = nullptr ; 
    Tran<double>* tr_m2w = nullptr ;
    Tran<double>* tr_w2m = nullptr ;
    // TODO: Tran is already (t,v,i) triplet : so can have just the one Tran 

    const char* ek = nullptr ; 
    const char* ev = nullptr ; 
    const char* ekvid = nullptr ; 

    sframe(); 
    sframe(const sframe& other); 
    ~sframe(); 
    void cleanup(); 

    sfr spawn_lite() const ; 

    void zero() ; 
    bool is_zero() const ; 

    std::string desc() const ; 

    static sframe Load( const char* dir, const char* name=NAME); 
    static sframe Load_(const char* path ); 
    static sframe Fabricate(float tx=0.f, float ty=0.f, float tz=0.f); 


    void set_grid(const std::vector<int>& cegs, float gridscale); 
    int ix0() const ; 
    int ix1() const ; 
    int iy0() const ; 
    int iy1() const ; 
    int iz0() const ; 
    int iz1() const ; 
    int num_photon() const ; 
    float gridscale() const ; 

    void set_ekv( const char* k ) ;  
    void set_ekv( const char* k, const char* v ) ;  
    const char* form_ekvid() const ; 
    const char* getFrameId() const ; 

    const char* get_frs() const ; // returns nullptr when frs is default  
    bool is_frs_default() const ; 

    const char* get_name() const ; // returns nullptr when frs is default  

    void set_midx_mord_gord(int midx, int mord, int gord); 
    int midx() const ; 
    int mord() const ; 
    int gord() const ; 

   


    void set_inst(int inst); 
    int inst() const ; 

    void set_identity(int ins, int gas, int sensor_identifier, int sensor_index );  // formerly set_ins_gas_ias
    int ins() const ; 
    int gas() const ; 
    int sensor_identifier() const ; 
    int sensor_index() const ; 

    void set_propagate_epsilon(float eps); 
    void set_hostside_simtrace(); 

    float propagate_epsilon() const ; 
    bool  is_hostside_simtrace() const ; 


    float* data() ; 
    const float* cdata() const ; 

    void write( float* dst, unsigned num_values ) const ;
    NP* getFrameArray() const ; 
    void save(const char* dir, const char* name=NAME) const ; 
    void save_extras(const char* dir) ;  // not const as may *prepare*


    void read( const float* src, unsigned num_values ) ; 
    void load(const char* dir, const char* name=NAME) ; 
    void load_(const char* path ) ; 
    void load(const NP* a) ; 



    void prepare();   // below are const by asserting that *prepare* has been called

    NP* transform_photon_m2w( const NP* ph, bool normalize=true ) const ; // hit OR photon (hmm could do record too)  
    NP* transform_photon_w2m( const NP* ph, bool normalize=true ) const ; 

    void transform_m2w( sphoton& p, bool normalize=true ) const ; 
    void transform_w2m( sphoton& p, bool normalize=true ) const ;


    Tran<double>* getTransform() const ; 


    void setTranslate(float x, float y, float z); 
    void setTransform(const qat4* m2w_ ); 

}; 


// ctor
inline sframe::sframe()
    :
    tr_m2w(nullptr),
    tr_w2m(nullptr)
{
#ifdef SFRAME_DEBUG
    printf("//sframe::sframe ctor NULL: tr_m2w %p tr_w2m %p \n", tr_m2w, tr_w2m ); 
#endif
}

/**
sframe copy ctor
--------------------

notes/issues/sframe_dtor_double_free_from_CSGOptiX__initFrame.rst

Note that a simple copy of non-nullptr transform pointers without the "prepare" 
would cause double ownership of tr_m2w tr_w2m and subsequent dtor double free errors.

**/

// copy ctor
inline sframe::sframe(const sframe& other)
{
    ce = other.ce ; 
    q1 = other.q1 ; 
    q2 = other.q2 ; 
    q3 = other.q3 ; 
    m2w = other.m2w ; 
    w2m = other.w2m ; 
    aux = other.aux ; 

    frs = other.frs ? strdup(other.frs) : nullptr ; 

    tr_m2w = nullptr ; 
    tr_w2m = nullptr ; 
    prepare();   // set the above from the m2w, w2m

    ek = other.ek ? strdup(other.ek) : nullptr ; 
    ev = other.ev ? strdup(other.ev) : nullptr ; 
    ekvid = other.ekvid ? strdup(other.ekvid) : nullptr ; 

#ifdef SFRAME_DEBUG
    printf("//sframe.copy.ctor NEW: tr_m2w %p tr_w2m %p \n", tr_m2w, tr_w2m ); 
#endif
}

/**
sframe::spawn_lite
-------------------

transitional 

**/

inline sfr sframe::spawn_lite() const 
{
    sfr l ; 
    l.ce.x = ce.x ; 
    l.ce.y = ce.y ; 
    l.ce.z = ce.z ; 
    l.ce.w = ce.w ;

    l.m2w = tr_m2w->t ;  
    l.w2m = tr_w2m->t ;  
 
    return l ; 
} 




/**
sframe::~sframe dtor
----------------------

**/
inline sframe::~sframe()
{
#ifdef SFRAME_DEBUG
    printf("//sframe::~sframe tr_m2w %p tr_w2m %p \n", tr_m2w, tr_w2m ); 
#endif
    //cleanup();  // JUST LEAKING FOR NOW : SEE PLAN IN notes/issues/sframe_dtor_double_free_from_CSGOptiX__initFrame.rst
}

inline void sframe::cleanup()
{
    free((void*)frs); 
    delete tr_m2w ; 
    delete tr_w2m ; 
    free((void*)ek);
    free((void*)ev);
    free((void*)ekvid); 
}


inline void sframe::zero()
{
    ce = {} ;   // 0
    q1 = {} ; 
    q2 = {} ; 
    q3 = {} ; 
    m2w = {} ;  // 1
    w2m = {} ;  // 2
    aux = {} ;  // 3

    frs = nullptr ; 
    tr_m2w = nullptr ;
    tr_w2m = nullptr ;
    ek = nullptr ; 
    ev = nullptr ; 
    ekvid = nullptr ; 
}


inline bool sframe::is_zero() const 
{
    return ce.x == 0. && ce.y == 0. && ce.z == 0. && ce.w == 0. ; 
}




inline std::string sframe::desc() const 
{
    std::stringstream ss ; 
    ss << "sframe::desc"
       << " inst " << inst() 
       << " frs " << ( frs ? frs : "-" ) << std::endl 
       << " ekvid " << ( ekvid ? ekvid : "-" )  
       << " ek " << ( ek ? ek : "-" )
       << " ev " << ( ev ? ev : "-" )
       << std::endl 
       << " ce  " << ce 
       << " is_zero " << is_zero() 
       << std::endl 
       << " m2w " << m2w 
       << std::endl 
       << " w2m " << w2m 
       << std::endl 
       << " midx " << std::setw(4) << midx()
       << " mord " << std::setw(4) << mord()
       << " gord " << std::setw(4) << gord()
       << std::endl 
       << " inst " << std::setw(4) << inst()
       << std::endl 
       << " ix0  " << std::setw(4) << ix0()
       << " ix1  " << std::setw(4) << ix1()
       << " iy0  " << std::setw(4) << iy0()
       << " iy1  " << std::setw(4) << iy1()
       << " iz0  " << std::setw(4) << iz0()
       << " iz1  " << std::setw(4) << iz1()
       << " num_photon " << std::setw(4) << num_photon()
       << std::endl 
       << " ins  " << std::setw(4) << ins()
       << " gas  " << std::setw(4) << gas()
       << " sensor_identifier  " << std::setw(7) << sensor_identifier()
       << " sensor_index  " << std::setw(5) << sensor_index()
       << std::endl 
       << " propagate_epsilon " << std::setw(10) << std::fixed << std::setprecision(5) << propagate_epsilon()
       << " is_hostside_simtrace " << ( is_hostside_simtrace() ? "YES" : "NO" ) 
       << std::endl 
       ;
    std::string str = ss.str(); 
    return str ; 
}


inline sframe sframe::Load(const char* dir, const char* name) // static
{
    sframe fr ; 
    fr.load(dir, name); 
    return fr ; 
}
inline sframe sframe::Load_(const char* path) // static
{
    sframe fr ; 
    fr.load_(path); 
    return fr ; 
}
/**
sframe::Fabricate
--------------------

Placeholder frame for testing, optionally with translation transform. 

**/
inline sframe sframe::Fabricate(float tx, float ty, float tz) // static
{
    sframe fr ; 
    fr.setTranslate(tx, ty, tz) ; 
    fr.prepare(); 
    return fr ; 
}







inline void sframe::set_grid(const std::vector<int>& cegs, float gridscale)
{
    assert( cegs.size() == 8 );   // use SFrameGenstep::StandardizeCEGS to convert 4/7/8 to 8   

    q1.i.x = cegs[0] ;  // ix0   these are after standardization
    q1.i.y = cegs[1] ;  // ix1
    q1.i.z = cegs[2] ;  // iy0 
    q1.i.w = cegs[3] ;  // iy1

    q2.i.x = cegs[4] ;  // iz0
    q2.i.y = cegs[5] ;  // iz1 
    q2.i.z = cegs[6] ;  // num_photon
    q2.f.w = gridscale ; 

    assert( cegs[7] == 1 ); // expecting 1 for cegs[7] other than fine regions where 2 (or 4) might be used

}

inline int sframe::ix0() const { return q1.i.x ; }
inline int sframe::ix1() const { return q1.i.y ; }
inline int sframe::iy0() const { return q1.i.z ; }
inline int sframe::iy1() const { return q1.i.w ; }
inline int sframe::iz0() const { return q2.i.x ; }
inline int sframe::iz1() const { return q2.i.y ; }
inline int sframe::num_photon() const { return q2.i.z ; }
inline float sframe::gridscale() const { return q2.f.w ; }


inline void sframe::set_ekv( const char* k ) 
{
    char* v = getenv(k) ; 
    set_ekv(k, v); 
}

inline void sframe::set_ekv( const char* k, const char* v ) 
{
    ek = k ? strdup(k) : nullptr ; 
    ev = v ? strdup(v) : nullptr ; 
    ekvid = form_ekvid(); 
}

inline const char* sframe::form_ekvid() const 
{
    std::stringstream ss ; 
    ss << "sframe_" 
       << ( ek ? ek : "ek" ) 
       << "_" 
       ; 
    for(int i=0 ; i < int(ev?strlen(ev):0) ; i++) ss << ( ev[i] == ':' ? '_' : ev[i] ) ; 
    std::string str = ss.str(); 
    return strdup(str.c_str()); 
}
inline const char* sframe::getFrameId() const 
{
    return ekvid ; 
}



inline const char* sframe::get_frs() const
{
    return is_frs_default() ? nullptr : frs ; 
}
inline bool sframe::is_frs_default() const 
{
    return frs != nullptr && strcmp(frs, DEFAULT_FRS) == 0 ; 
}
inline const char* sframe::get_name() const 
{
    const char* f = get_frs(); 
    return f ? f : DEFAULT_NAME ; 
}

inline void sframe::set_midx_mord_gord(int midx, int mord, int gord)
{
    q3.i.x = midx ; 
    q3.i.y = mord ; 
    q3.i.z = gord ; 
}
inline int sframe::midx() const { return q3.i.x ; }
inline int sframe::mord() const { return q3.i.y ; }
inline int sframe::gord() const { return q3.i.z ; }
inline int sframe::inst() const { return q3.i.w ; }

inline void sframe::set_inst(int inst){ q3.i.w = inst ; }

inline void sframe::set_identity(int ins, int gas, int sensor_identifier, int sensor_index ) // formerly set_ins_gas_ias
{
    aux.q0.i.x = ins ; 
    aux.q0.i.y = gas ; 
    aux.q0.i.z = sensor_identifier ; 
    aux.q0.i.w = sensor_index  ; 
}
inline int sframe::ins() const { return aux.q0.i.x ; }
inline int sframe::gas() const { return aux.q0.i.y ; }
inline int sframe::sensor_identifier() const { return aux.q0.i.z ; }
inline int sframe::sensor_index() const {      return aux.q0.i.w ; }


inline void sframe::set_propagate_epsilon(float eps){     aux.q1.f.x = eps ; }
inline void sframe::set_hostside_simtrace(){              aux.q1.u.y = 1u ; }

inline float sframe::propagate_epsilon() const   { return aux.q1.f.x ; }
inline bool sframe::is_hostside_simtrace() const { return aux.q1.u.y == 1u ; } 


inline const float* sframe::cdata() const 
{
    return (const float*)&ce.x ;  
}
inline float* sframe::data()  
{
    return (float*)&ce.x ;  
}
inline void sframe::write( float* dst, unsigned num_values ) const 
{
    assert( num_values == NUM_VALUES ); 
    char* dst_bytes = (char*)dst ; 
    char* src_bytes = (char*)cdata(); 
    unsigned num_bytes = sizeof(float)*num_values ; 
    memcpy( dst_bytes, src_bytes, num_bytes );
}    

inline void sframe::read( const float* src, unsigned num_values ) 
{
    assert( num_values == NUM_VALUES ); 
    char* src_bytes = (char*)src ; 
    char* dst_bytes = (char*)data(); 
    unsigned num_bytes = sizeof(float)*num_values ; 
    memcpy( dst_bytes, src_bytes, num_bytes );
}    

inline NP* sframe::getFrameArray() const 
{
    NP* a = NP::Make<float>(NUM_4x4, 4, 4) ; 
    write( a->values<float>(), NUM_4x4*4*4 ) ; 

    a->set_meta<std::string>("creator", "sframe::getFrameArray"); 
    if(frs) a->set_meta<std::string>("frs", frs); 
    if(ek) a->set_meta<std::string>("ek", ek); 
    if(ev) a->set_meta<std::string>("ev", ev); 
    if(ekvid) a->set_meta<std::string>("ekvid", ekvid); 

    return a ; 
}
inline void sframe::save(const char* dir, const char* name_ ) const
{
    if(VERBOSE) std::cout 
        << "[ sframe::save " 
        << " dir : " << ( dir ? dir : "MISSING_DIR" ) 
        << " name: " << ( name_ ? name_ : "MISSING_NAME" ) 
        << std::endl 
        ; 

    std::string name = U::form_name( name_ , ".npy" ) ; 
    NP* a = getFrameArray(); 

    a->save(dir, name.c_str()); 

    if(VERBOSE) std::cout 
       << "] sframe::save "
       << std::endl
       ;

}
inline void sframe::save_extras(const char* dir)
{
    if(tr_m2w == nullptr) prepare(); 
    tr_m2w->save(dir, "m2w.npy");
    tr_w2m->save(dir, "w2m.npy");
}


inline void sframe::load(const char* dir, const char* name_ ) 
{
    std::string name = U::form_name( name_ , ".npy" ) ; 
    const NP* a = NP::Load(dir, name.c_str() ); 
    load(a); 
}
inline void sframe::load_(const char* path_)   // eg $A_FOLD/sframe.npy
{
    const NP* a = NP::Load(path_);
    if(!a) std::cerr 
       << "sframe::load_ ERROR : non-existing" 
       << " path_ " << path_ 
       << std::endl   
       ;
    assert(a); 
    load(a); 
}
inline void sframe::load(const NP* a) 
{
    read( a->cvalues<float>() , NUM_VALUES );   
    std::string _frs = a->get_meta<std::string>("frs", ""); 
    if(!_frs.empty()) frs = strdup(_frs.c_str()); 
}



/**
sframe::prepare
-----------------


**/

inline void sframe::prepare()
{
    tr_m2w = Tran<double>::ConvertFromQat(&m2w) ;
    tr_w2m = Tran<double>::ConvertFromQat(&w2m) ;
}


/**
sframe::transform_photon_m2w
-------------------------------

Canonical call from SEvt::setFrame for transforming input photons into frame 
When normalize is true the mom and pol are normalized after the transformation. 

Note that the transformed photon array is always in double precision. 
That will be narrowed down to float prior to upload by QEvent::setInputPhoton

**/

inline NP* sframe::transform_photon_m2w( const NP* ph, bool normalize ) const 
{
    if( ph == nullptr ) return nullptr ; 
    if(!tr_m2w) std::cerr << "sframe::transform_photon_m2w MUST sframe::prepare before calling this " << std::endl; 
    assert( tr_m2w) ; 
    NP* pht = Tran<double>::PhotonTransform(ph, normalize,  tr_m2w );
    assert( pht->ebyte == 8 ); 
    return pht ; 
}

inline NP* sframe::transform_photon_w2m( const NP* ph, bool normalize  ) const 
{
    if( ph == nullptr ) return nullptr ; 
    if(!tr_w2m) std::cerr << "sframe::transform_photon_w2m MUST sframe::prepare before calling this " << std::endl; 
    assert( tr_w2m ) ; 
    NP* pht = Tran<double>::PhotonTransform(ph, normalize, tr_w2m );
    assert( pht->ebyte == 8 ); 
    return pht ; 
}

inline void sframe::transform_m2w( sphoton& p, bool normalize ) const 
{
    if(!tr_m2w) std::cerr << "sframe::transform_m2w MUST sframe::prepare before calling this " << std::endl; 
    assert( tr_m2w) ;
    p.transform( tr_m2w->t, normalize ); 
}

inline void sframe::transform_w2m( sphoton& p, bool normalize ) const 
{
    if(!tr_w2m) std::cerr << "sframe::transform_w2m MUST sframe::prepare before calling this " << std::endl; 
    assert( tr_w2m) ;
    p.transform( tr_w2m->t, normalize ); 
}


inline Tran<double>* sframe::getTransform() const 
{
    double eps = 1e-3 ; // formerly used 1e-6 gives idenity check warnings  
    Tran<double>* geotran = Tran<double>::FromPair( &m2w, &w2m, eps ); 
    return geotran ; 
}

inline void sframe::setTranslate(float x, float y, float z)  // UNTESTED
{
    qat4 m2w(x,y,z); 
    setTransform(&m2w); 
}
inline void sframe::setTransform(const qat4* m2w_ )  // UNTESTED
{
    const qat4* w2m_ = Tran<double>::Invert(m2w_);   
    qat4::copy(m2w, *m2w_ );  
    qat4::copy(w2m, *w2m_ );  
}

inline std::ostream& operator<<(std::ostream& os, const sframe& fr)
{
    os << fr.desc() ; 
    return os; 
}


