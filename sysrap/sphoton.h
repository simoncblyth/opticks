#pragma once
/**
sphoton.h
============

+----+----------------+----------------+----------------+----------------+--------------------------+
| q  |      x         |      y         |     z          |      w         |  notes                   |
+====+================+================+================+================+==========================+
|    |  pos.x         |  pos.y         |  pos.z         |  time          |                          |
| q0 |                |                |                |                |                          |
|    |                |                |                |                |                          |
+----+----------------+----------------+----------------+----------------+--------------------------+
|    |  mom.x         |  mom.y         | mom.z          |  iindex        |                          |
| q1 |                |                |                | (unsigned)     |                          |
|    |                |                |                |                |                          |
+----+----------------+----------------+----------------+----------------+--------------------------+
|    |  pol.x         |  pol.y         |  pol.z         |  wavelength    |                          |
| q2 |                |                |                |                |                          |
|    |                |                |                |                |                          |
+----+----------------+----------------+----------------+----------------+--------------------------+
|    | boundary_flag  |  identity      |  orient_idx    |  flagmask      |  (unsigned)              |
| q3 | (3,0)          |                |  orient:1bit   |                |                          |
|    |                |                |  (3,2)         |                |                          |
+----+----------------+----------------+----------------+----------------+--------------------------+


iindex
    instance index of intersected geometry
    (see squad.h:quad2 for details)

boundary (16 bit)
    boundary index of intersected geometry
    index corresponds to unique combinations of four material and surface indices
    (omat,osur,isur,imat)

flag (16 bit)
    OpticksPhoton.h history flag enum eg: TO CK SI BT BR SR AB RE ...

identity (32 bit)
    currently sensor_identfier (see squad.h:quad2 for details)
    Formerly was combination of primIdx and instanceId of intersected geometry.

orient (1 bit)
    set according to the sign of cosTheta
    (dot product of surface normal and momentum at last intersect)

idx (31 bit)
    photon index, always exists even before any intersect

flagmask (32 bit)
    bitwise-OR of step point flag enumeration
    (see sysrap/OpticksPhoton.h sysrap/OpticksPhoton.hh for details)
    Always exists even before any intersect with the generation flag CK/SI/TO.



NB locations and packing here need to match ana/p.py
------------------------------------------------------

See ana/p.py for python accessors such as::

    boundary_  = lambda p:p.view(np.uint32)[3,0] >> 16
    flag_      = lambda p:p.view(np.uint32)[3,0] & 0xffff

    identity_ = lambda p:p.view(np.uint32)[3,1]
    primIdx_   = lambda p:identity_(p) >> 16
    instanceId_  = lambda p:identity_(p) & 0xffff

    idx_      = lambda p:p.view(np.uint32)[3,2] & 0x7fffffff
    orient_   = lambda p:p.view(np.uint32)[3,2] >> 31

    flagmask_ = lambda p:p.view(np.uint32)[3,3]



POSSIBLE FLAGS REARRANGEMENT
------------------------------

::

    unsigned idx ;
    unsigned orient_identity ;
    unsigned boundary_flag ;
    unsigned flagmask ;

idx always exists for a photon,
BUT: orient is only set on intersects together with boundary and identity
(plus flag but that is also be set when no intersect)

SO IT WOULD BE BETTER TO HAVE orient_identity and idx alone if identity can spare one bit ?

    identity = (( prim_idx & 0xffff ) << 16 ) | ( instance_id & 0xffff ) ;

JUNO max prim_idx ~3245 : so thats OK

    In [1]: 0xffff
    Out[1]: 65535

    In [2]: 0x7fff
    Out[2]: 32767


**/



#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SPHOTON_METHOD __host__ __device__ __forceinline__
#else
#    define SPHOTON_METHOD inline
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
   #include <iostream>
   #include <iomanip>
   #include <sstream>
   #include <vector>
   #include <cstring>
   #include <cassert>
   #include <glm/glm.hpp>

   #include "scuda.h"
   #include "stran.h"
   #include "smath.h"
   #include "OpticksPhoton.h"

   struct NP ;
#endif

struct sphoton
{
    float3 pos ;        // 0
    float  time ;

    float3 mom ;        // 1
    unsigned iindex ;   // instance index, t.record[:,:,1,3].view(np.int32)

    float3 pol ;         // 2
    float  wavelength ;

    unsigned boundary_flag ;  // 3
    unsigned identity ;       // [:,3,1]
    unsigned orient_idx ;
    unsigned flagmask ;

    SPHOTON_METHOD void set_prd( unsigned  boundary, unsigned  identity, float  orient, unsigned iindex );

    SPHOTON_METHOD unsigned idx() const {      return orient_idx & 0x7fffffffu  ;  }
    SPHOTON_METHOD float    orient() const {   return ( orient_idx & 0x80000000u ) ? -1.f : 1.f ; }

    SPHOTON_METHOD void set_orient(float orient){ orient_idx = ( orient_idx & 0x7fffffffu ) | (( orient < 0.f ? 0x1 : 0x0 ) << 31 ) ; } // clear orient bit and then set it
    SPHOTON_METHOD void set_idx( unsigned idx ){  orient_idx = ( orient_idx & 0x80000000u ) | ( 0x7fffffffu & idx ) ; }   // retain bit 31 asis

    SPHOTON_METHOD unsigned flag() const {     return boundary_flag & 0xffffu ; } // flag___     = lambda p:p.view(np.uint32)[...,3,0] & 0xffff
    SPHOTON_METHOD unsigned boundary() const { return boundary_flag >> 16 ; }     // boundary___ = lambda p:p.view(np.uint32)[...,3,0] >> 16

    SPHOTON_METHOD void     set_flag(unsigned flag) {         boundary_flag = ( boundary_flag & 0xffff0000u ) | ( flag & 0xffffu ) ; flagmask |= flag ;  } // clear flag bits then set them
    SPHOTON_METHOD void     set_boundary(unsigned boundary) { boundary_flag = ( boundary_flag & 0x0000ffffu ) | (( boundary & 0xffffu ) << 16 ) ; }        // clear boundary bits then set them

    SPHOTON_METHOD void     addto_flagmask(unsigned flag) { flagmask |=  flag ; }
    SPHOTON_METHOD void     scrub_flagmask(unsigned flag) { flagmask &= ~flag ; }

    SPHOTON_METHOD void zero_flags() { boundary_flag = 0u ; identity = 0u ; orient_idx = 0u ; flagmask = 0u ; }

    SPHOTON_METHOD float* data() {               return &pos.x ; }
    SPHOTON_METHOD const float* cdata() const {  return &pos.x ; }




    SPHOTON_METHOD void zero()
    {
       pos.x = 0.f ; pos.y = 0.f ; pos.z = 0.f ; time = 0.f ;
       mom.x = 0.f ; mom.y = 0.f ; mom.z = 0.f ; iindex = 0u ;
       pol.x = 0.f ; pol.y = 0.f ; pol.z = 0.f ; wavelength = 0.f ;
       boundary_flag = 0u ; identity = 0u ; orient_idx = 0u ; flagmask = 0u ;
    }

#if defined(__CUDACC__) || defined(__CUDABE__)
#else

    SPHOTON_METHOD bool is_cerenkov()  const { return (flagmask & CERENKOV) != 0 ; }
    SPHOTON_METHOD bool is_reemit() const {    return (flagmask & BULK_REEMIT) != 0 ; }

    SPHOTON_METHOD unsigned flagmask_count() const ;
    SPHOTON_METHOD std::string desc() const ;
    SPHOTON_METHOD std::string descBase() const ;
    SPHOTON_METHOD std::string descPos() const ;
    SPHOTON_METHOD std::string descDir() const ;
    SPHOTON_METHOD std::string descDetail() const ;
    SPHOTON_METHOD std::string descFlag() const ;
    SPHOTON_METHOD std::string descDigest() const ;

    SPHOTON_METHOD const char* flag_() const ;
    SPHOTON_METHOD const char* abbrev_() const ;
    SPHOTON_METHOD std::string flagmask_() const ;


    SPHOTON_METHOD void ephoton() ;
    SPHOTON_METHOD void normalize_mom_pol();
    SPHOTON_METHOD void transverse_mom_pol();
    SPHOTON_METHOD void set_polarization(float frac_twopi);
    SPHOTON_METHOD static sphoton make_ephoton();
    SPHOTON_METHOD static NP* make_ephoton_array(int num_photon);
    SPHOTON_METHOD std::string digest(unsigned numval=16) const  ;
    SPHOTON_METHOD static bool digest_match( const sphoton& a, const sphoton& b, unsigned numval=16 ) ;

    SPHOTON_METHOD static float  DeltaMax( const float* aa, const float* bb, int num );
    SPHOTON_METHOD static float4 DeltaMax( const sphoton& a, const sphoton& b ) ;
    SPHOTON_METHOD static bool EqualFlags( const sphoton& a, const sphoton& b) ;

    SPHOTON_METHOD static void Get( sphoton& p, const NP* a, unsigned idx );
    SPHOTON_METHOD static void Get( std::vector<sphoton>& pp, const NP* a );

    SPHOTON_METHOD void transform_float( const glm::tmat4x4<float>&  tr, bool normalize=true );  // widens transform and uses below
    SPHOTON_METHOD void transform(       const glm::tmat4x4<double>& tr, bool normalize=true );
#endif

};


/**
sphoton::set_prd
-----------------

This is canonically invoked GPU side by qsim::propagate
copying geometry info from the quad2 PRD struct into the sphoton.

TODO: relocate identity - 1 offsetting into here as this
marks the transition from geometry to event information
and would allow the offsetting to be better hidden.

See ~/opticks/notes/issues/sensor_identifier_offset_by_one_wrinkle.rst

**/


SPHOTON_METHOD void sphoton::set_prd( unsigned  boundary_, unsigned  identity_, float  orient_, unsigned iindex_ )
{
    set_boundary(boundary_);
    identity = identity_ ;
    set_orient( orient_ );
    iindex = iindex_ ;
}



#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#include <cassert>
#include <bitset>
#include "sdigest.h"
#include "OpticksPhoton.hh"
#include "NP.hh"

SPHOTON_METHOD unsigned sphoton::flagmask_count() const
{
    return std::bitset<32>(flagmask).count() ;   // NB counting bits, not nibbles with bits
}

SPHOTON_METHOD std::string sphoton::desc() const
{
    std::stringstream ss ;
    ss << descBase() << " "<< descDetail() ;
    std::string s = ss.str();
    return s ;
}

SPHOTON_METHOD std::string sphoton::descPos() const
{
    std::stringstream ss ;
    ss
        << " pos " << pos
        << " t  " << std::setw(8) << time
        << " "
        ;
    std::string s = ss.str();
    return s ;
}

SPHOTON_METHOD std::string sphoton::descDir() const
{
    std::stringstream ss ;
    ss
        << " mom " << mom
        << " iindex " << std::setw(4) << iindex
        << " "
        << " pol " << pol
        << " wl " << std::setw(8) << wavelength
        << " "
        ;
    std::string s = ss.str();
    return s ;
}

SPHOTON_METHOD std::string sphoton::descBase() const
{
    std::stringstream ss ;
    ss  << descPos()
        << descDir()
        ;
    std::string s = ss.str();
    return s ;
}
SPHOTON_METHOD std::string sphoton::descDetail() const
{
    std::stringstream ss ;
    ss
        << " bn " << boundary()
        << " fl " << std::hex << flag() << std::dec
        << " id " << identity
        << " or " << orient()
        << " ix " << idx()
        << " fm " << std::hex << flagmask  << std::dec
        << " ab " << OpticksPhoton::Abbrev( flag() )
        << " ii " << iindex
        ;
    std::string s = ss.str();
    return s ;
}

SPHOTON_METHOD std::string sphoton::descDigest() const
{
    std::stringstream ss ;
    ss
        << " "
        << " digest(16) " << digest(16)
        << " "
        << " digest(12) " << digest(12)
        ;
    std::string s = ss.str();
    return s ;
}


SPHOTON_METHOD const char* sphoton::flag_() const
{
    return OpticksPhoton::Flag(flag()) ;
}

SPHOTON_METHOD const char* sphoton::abbrev_() const
{
    return OpticksPhoton::Abbrev(flag()) ;
}
SPHOTON_METHOD std::string sphoton::flagmask_() const
{
    return OpticksPhoton::FlagMaskLabel(flagmask) ;
}

SPHOTON_METHOD std::string sphoton::descFlag() const
{
    std::stringstream ss ;
    ss
       << " sphoton idx " << idx()
       << " flag " << flag_()
       << " flagmask " << flagmask_()
       ;
    std::string s = ss.str();
    return s ;
}


/**
sphoton::ephoton
------------------

*ephoton* is used from qudarap/tests/QSimTest generate tests as the initial photon,
which gets persisted to p0.npy
The script qudarap/tests/ephoton.sh sets the envvars defining the photon
depending on the TEST envvar.

**/

SPHOTON_METHOD void sphoton::ephoton()
{
    quad4& q = (quad4&)(*this) ;

    qvals( q.q0.f ,        "EPHOTON_POST" , "0,0,0,0" );                      // position, time
    qvals( q.q1.f, q.q2.f, "EPHOTON_MOMW_POLW", "1,0,0,1,0,1,0,500" );  // direction, weight,  polarization, wavelength
    qvals( q.q3.i ,        "EPHOTON_FLAG", "0,0,0,0" );
    normalize_mom_pol();
    transverse_mom_pol();
}

SPHOTON_METHOD void sphoton::normalize_mom_pol()
{
    mom = normalize(mom);
    pol = normalize(pol);
}

SPHOTON_METHOD void sphoton::transverse_mom_pol()
{
    float mom_pol = fabsf( dot(mom, pol)) ;
    float eps = 1e-5 ;
    bool is_transverse = mom_pol < eps ;

    if(!is_transverse )
    {
        std::cout
             << " sphoton::transverse_mom_pol "
             << " FAIL "
             << " mom " << mom
             << " pol " << pol
             << " mom_pol " << mom_pol
             << " eps " << eps
             << " is_transverse " << is_transverse
             << std::endl
             ;
    }
    assert(is_transverse);
}


/**
sphoton::set_polarization
---------------------------

Start with a vector in XY plane and rotate that using smath::rotateUz
to be transverse to mom.

**/

SPHOTON_METHOD void sphoton::set_polarization(float frac_twopi)
{
    float phase = 2.f*M_PIf*frac_twopi ;
    pol.x = cosf(phase) ;
    pol.y = sinf(phase) ;
    pol.z = 0.f ;
    smath::rotateUz(pol, mom); // rotate pol to be transverse to mom
}


SPHOTON_METHOD sphoton sphoton::make_ephoton()  // static
{
    sphoton p ;
    p.ephoton();
    return p ;
}

/**
sphoton::make_ephoton_array
-----------------------------

Formerly QSim::duplicate_dbg_ephoton

This is called from QSimTest::mock_propagate

**/

SPHOTON_METHOD NP* sphoton::make_ephoton_array(int num_photon) // static
{
    sphoton ep ;
    ep.ephoton() ;

    NP* ph = NP::Make<float>(num_photon, 4, 4 );
    sphoton* pp  = (sphoton*)ph->bytes();
    for(int i=0 ; i < num_photon ; i++)
    {
        sphoton p = ep  ;  // start from ephoton
        p.pos.y = float(i)*100.f ;
        pp[i] = p ;
    }
    return ph ;
}



SPHOTON_METHOD std::string sphoton::digest(unsigned numval) const
{
    assert( numval <= 16 );
    return sdigest::Buf( (const char*)cdata() , numval*sizeof(float) );
}

SPHOTON_METHOD bool sphoton::digest_match( const sphoton& a, const sphoton& b, unsigned numval )  // static
{
    std::string adig = a.digest(numval);
    std::string bdig = b.digest(numval);
    return strcmp( adig.c_str(), bdig.c_str() ) == 0 ;
}

SPHOTON_METHOD float sphoton::DeltaMax( const float* aa, const float* bb, int num )  // static
{
    float dmax = 0.f ;
    for(int i=0 ; i < num ; i++)
    {
        float a = *(aa+i) ;
        float b = *(bb+i) ;
        float ab = std::abs(a - b);
        if(ab > dmax) dmax = ab ;
    }
    return dmax ;
}

SPHOTON_METHOD float4 sphoton::DeltaMax( const sphoton& a, const sphoton& b )  // static
{
    float4 dmax = make_float4( 0.f, 0.f, 0.f, 0.f );
    dmax.x = DeltaMax( &a.pos.x, &b.pos.x, 3 );
    dmax.y = DeltaMax( &a.mom.x, &b.mom.x, 3 );
    dmax.z = DeltaMax( &a.pol.x, &b.pol.x, 3 );
    dmax.w = std::max( DeltaMax( &a.time,  &b.time, 1 ), DeltaMax( &a.wavelength,  &b.wavelength, 1 ) ) ;
    return dmax ;
}


SPHOTON_METHOD bool sphoton::EqualFlags( const sphoton& a, const sphoton& b) // static
{
    return a.iindex == b.iindex &&
           a.boundary_flag == b.boundary_flag &&
           a.identity == b.identity &&
           a.orient_idx == b.orient_idx &&
           a.flagmask == b.flagmask
           ;

}




SPHOTON_METHOD void sphoton::Get( sphoton& p, const NP* a, unsigned idx )
{
    bool expected = a && a->has_shape(-1,4,4) && a->ebyte == sizeof(float) && idx < unsigned(a->shape[0]) ;
    if(!expected) std::cerr
        << "sphoton::Get not expected error "
        << " a " << ( a ? "Y" : "N" )
        << " a.shape " << ( a ? a->sstr() : "-" )
        << " a.ebyte " << ( a ? a->ebyte : -1 )
        << " a.shape[0] " << ( a ? a->shape[0] : -1 )
        << " idx " << idx
        << std::endl
        ;

    assert( expected  );
    assert( sizeof(sphoton) == sizeof(float)*16 );
    memcpy( &p, a->cvalues<float>() + idx*16, sizeof(sphoton) );
}

SPHOTON_METHOD void sphoton::Get( std::vector<sphoton>& pp, const NP* a )
{
    assert(a && a->has_shape(-1,4,4) && a->ebyte == sizeof(float) );
    assert( sizeof(sphoton) == sizeof(float)*16 );
    unsigned num = a->shape[0] ;
    pp.resize(num);
    memcpy( pp.data(), a->cvalues<float>(), sizeof(sphoton)*num );
}




/**
sphoton::transform_float
--------------------------------

Its better to keep transforms in double and use sphoton::transform but if
double precision transforms are not yet available this will widen
the transform and use that, which is better than using float.

**/

SPHOTON_METHOD void sphoton::transform_float( const glm::tmat4x4<float>& tr, bool normalize )
{
    glm::tmat4x4<double> trd ;
    TranConvert(trd, tr);
    transform(trd, normalize );
}

/**
sphoton::transform
--------------------

Applies a double precision transform to float precision sphoton
with Tran<double>::ApplyToFloat by temporarily widening the sphoton
pos/mom/pol in order to do the matrix multiplication in double precision
and then save back into the sphoton in float.

And alternative to this would be to widen the sphoton to sphotond
and then use that.

**/

SPHOTON_METHOD void sphoton::transform( const glm::tmat4x4<double>& tr, bool normalize )
{
    float one(1.);
    float zero(0.);

    unsigned count = 1 ;
    unsigned stride = 4*4 ; // effectively not used as count is 1

    assert( sizeof(*this) == sizeof(float)*16 );
    float* p0 = (float*)this ;

    Tran<double>::ApplyToFloat( tr, p0, one,  count, stride, 0, false );      // transform pos as position
    Tran<double>::ApplyToFloat( tr, p0, zero, count, stride, 4, normalize );  // transform mom as direction
    Tran<double>::ApplyToFloat( tr, p0, zero, count, stride, 8, normalize );  // transform pol as direction
}



#endif


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
inline std::ostream& operator<<(std::ostream& os, const sphoton& p )
{
    os << "sphoton"
       << std::endl
       << p.desc()
       << std::endl
       ;
    return os;
}
#endif



union qphoton
{
    quad4   q ;
    sphoton p ;
};


struct sphoton_selector
{
    unsigned hitmask ;
    sphoton_selector(unsigned hitmask_) : hitmask(hitmask_) {};
    SPHOTON_METHOD bool operator() (const sphoton& p) const { return ( p.flagmask  & hitmask ) == hitmask  ; }   // require all bits of the mask to be set
    SPHOTON_METHOD bool operator() (const sphoton* p) const { return ( p->flagmask & hitmask ) == hitmask  ; }   // require all bits of the mask to be set
};



#if defined(__CUDACC__) || defined(__CUDABE__)
#else

struct sphotond
{
    double3 pos ;
    double  time ;

    double3 mom ;
    unsigned long long iindex ;  // formerly float weight, but have never used that

    double3 pol ;
    double  wavelength ;

    unsigned long long boundary_flag ;
    unsigned long long identity ;
    unsigned long long orient_idx ;
    unsigned long long flagmask ;


   SPHOTON_METHOD static void FromFloat( sphotond& d, const sphoton& s );
   SPHOTON_METHOD static void Get( sphotond& p, const NP* a, unsigned idx );
   SPHOTON_METHOD void transform_float( const glm::tmat4x4<float>& tr,  bool normalize=true );
   SPHOTON_METHOD void transform(       const glm::tmat4x4<double>& tr, bool normalize=true );


};


SPHOTON_METHOD void sphotond::FromFloat( sphotond& d, const sphoton& s )
{
    typedef unsigned long long ull ;

    d.pos.x = double(s.pos.x) ;
    d.pos.y = double(s.pos.y) ;
    d.pos.z = double(s.pos.z) ;
    d.time  = double(s.time) ;

    d.mom.x = double(s.mom.x) ;
    d.mom.y = double(s.mom.y) ;
    d.mom.z = double(s.mom.z) ;
    d.iindex = ull(s.iindex) ;

    d.pol.x = double(s.pol.x) ;
    d.pol.y = double(s.pol.y) ;
    d.pol.z = double(s.pol.z) ;
    d.wavelength  = double(s.wavelength) ;

    d.boundary_flag = ull(s.boundary_flag) ;
    d.identity      = ull(s.identity) ;
    d.orient_idx    = ull(s.orient_idx) ;
    d.flagmask      = ull(s.flagmask) ;
}

SPHOTON_METHOD void sphotond::Get( sphotond& p, const NP* a, unsigned idx )
{
    assert(a && a->has_shape(-1,4,4) && a->ebyte == sizeof(double) && idx < unsigned(a->shape[0]) );
    assert( sizeof(sphotond) == sizeof(double)*16 );
    memcpy( &p, a->cvalues<double>() + idx*16, sizeof(sphotond) );
}

SPHOTON_METHOD void sphotond::transform_float( const glm::tmat4x4<float>& tr, bool normalize )
{
    glm::tmat4x4<double> trd ;
    TranConvert(trd, tr);
    transform(trd, normalize );
}

SPHOTON_METHOD void sphotond::transform( const glm::tmat4x4<double>& tr, bool normalize )
{
    double one(1.);
    double zero(0.);

    unsigned count = 1 ;
    unsigned stride = 4*4 ; // effectively not used as count is 1

    assert( sizeof(*this) == sizeof(double)*16 );
    double* p0 = (double*)this ;

    Tran<double>::Apply( tr, p0, one,  count, stride, 0, false );      // transform pos as position
    Tran<double>::Apply( tr, p0, zero, count, stride, 4, normalize );  // transform mom as direction
    Tran<double>::Apply( tr, p0, zero, count, stride, 8, normalize );  // transform pol as direction
}

#endif




