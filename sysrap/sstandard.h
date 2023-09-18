#pragma once
/**
sstandard.h : standard domain arrays
========================================

This houses standardized "mat" and "sur" arrays with  
material/surface properties interpolated onto a standard 
energy/wavelength domain. 

mat
    populated by U4Tree::initMaterials using U4Material::MakeStandardArray 
    aiming to replace X4/GGeo workflow GMaterialLib buffer.
    The shape of the mat array::

      (~20:num_mat, 2:num_payload_cat, num_wavelength_samples, 4:payload_values ) 

sur
    populated by U4Tree::initSurfaces_Serialize using U4SurfaceArray.h 
    aiming to replace X4/GGeo workflow GSurfaceLib buffer

bnd
    interleaved mat and sur array items 
    Example shapes::

        mat    (20, 2, 761, 4)
        sur    (40, 2, 761, 4)
        bnd (53, 4, 2, 761, 4)

    The (2,761,4) is considered the payload comprising 
    two payload groups and payload quad at 761 interpolated 
    domain energies/wavelenths. 
    This payload shape is used to allow the bnd values 
    to be accessed via a float4 GPU texture. 

bd 
   (omat,osur,isur,imat) int pointers  

optical
   material and surface param 

wavelength
   from sdomain::get_wavelength_nm

energy
   from sdomain::get_energy_eV 

rayleigh
   populated by U4Tree::initRayleigh
   



In the old X4/GGeo workflow, the bnd buffer was created with::

    GBndLib::createBufferForTex2d
    -------------------------------

    GBndLib double buffer is a memcpy zip of the MaterialLib and SurfaceLib buffers
    pulling together data based on the indices for the materials and surfaces 
    from the m_bnd guint4 buffer

    Typical dimensions : (128, 4, 2, 39, 4)   

               128 : boundaries, 
                 4 : mat-or-sur for each boundary  
                 2 : payload-categories corresponding to NUM_FLOAT4
                39 : wavelength samples
                 4 : double4-values

    The only dimension that can easily be extended is the middle payload-categories one, 
    the low side is constrained by layout needed to OptiX tex2d<float4> as this 
    buffer is memcpy into the texture buffer
    high side is constained by not wanting to change texture line indices 

    The 39 wavelength samples is historical. There is a way to increase this
    to 1nm FINE_DOMAIN binning.

**/

#include <limits>
#include <array>

#include "NPFold.h"
#include "NPX.h"
#include "sproplist.h"
#include "sdomain.h"
#include "smatsur.h"
#include "snam.h"

struct sstandard
{ 
    static constexpr const bool VERBOSE = false ; 
    static constexpr const char* IMPLICIT_PREFIX = "Implicit_RINDEX_NoRINDEX" ;
    const sdomain* dom ; 

    const NP* wavelength ; 
    const NP* energy ; 
    const NP* rayleigh ; 
    const NP* mat ; 
    const NP* sur ; 
    const NP* bd ; 
    const NP* bnd ; 
    const NP* optical ;  

    const NP* icdf ; 
   

    sstandard(); 

    void deferred_init(
        const std::vector<int4>& vbd,
        const std::vector<std::string>& bdname,
        const std::vector<std::string>& suname,
        const NPFold* surface
    );

    NPFold* serialize() const ; 
    void import(const NPFold* fold ); 

    void save(const char* base, const char* rel ); 
    void load(const char* base, const char* rel ); 


    static NP* make_bd(
        const std::vector<int4>& vbd, 
        const std::vector<std::string>& bdname
    ); 

    static NP* make_optical(
        const std::vector<int4>& vbd, 
        const std::vector<std::string>& suname, 
        const NPFold* surface 
    ); 

    static NP* make_bnd(
        const std::vector<int4>& vbd, 
        const std::vector<std::string>& bdname, 
        const NP* mat, 
        const NP* sur
    );
 
    static void column_range(int4& mn, int4& mx,  const std::vector<int4>& vbd) ; 
    static NP* unused_mat(const std::vector<std::string>& names, const NPFold* fold ); 
    static NP* unused_sur(const std::vector<std::string>& names, const NPFold* fold ); 
    static NP* unused_create(const sproplist* pl,  const std::vector<std::string>& names, const NPFold* fold ); 
};


inline sstandard::sstandard()
    :
    dom(nullptr),
    wavelength(nullptr),
    energy(nullptr),
    rayleigh(nullptr),
    mat(nullptr),
    sur(nullptr), 
    bd(nullptr), 
    bnd(nullptr), 
    optical(nullptr),
    icdf(nullptr)
{
}


/**
sstandard::deferred_init
--------------------------

NB deferred init called from stree::initStandard 
after mat and sur have been filled. 

**/

inline void sstandard::deferred_init(
        const std::vector<int4>& vbd,
        const std::vector<std::string>& bdname,
        const std::vector<std::string>& suname,
        const NPFold* surface
    )   
{
    dom = new sdomain ; 

    wavelength = dom->get_wavelength_nm() ; 
    energy = dom->get_energy_eV() ; 

    bd      = make_bd(     vbd, bdname ); 
    bnd     = make_bnd(    vbd, bdname, mat, sur ) ; 
    optical = make_optical(vbd, suname, surface) ; 
}


inline NPFold* sstandard::serialize() const 
{
    NPFold* fold = new NPFold ; 

    fold->add(snam::WAVELENGTH , wavelength ); 
    fold->add(snam::ENERGY,      energy ); 

    fold->add(snam::RAYLEIGH,    rayleigh ); 
    fold->add(snam::MAT ,    mat ); 
    fold->add(snam::SUR ,    sur ); 

    fold->add(snam::BD,      bd ); 
    fold->add(snam::BND,     bnd ); 
    fold->add(snam::OPTICAL, optical );  

    fold->add(snam::ICDF, icdf) ; 

    return fold ; 
}

inline void sstandard::import(const NPFold* fold )
{
    wavelength = fold->get(snam::WAVELENGTH); 
    energy = fold->get(snam::ENERGY); 

    rayleigh = fold->get(snam::RAYLEIGH); 
    mat = fold->get(snam::MAT); 
    sur = fold->get(snam::SUR); 

    bd = fold->get(snam::BD); 
    bnd = fold->get(snam::BND); 
    optical = fold->get(snam::OPTICAL); 

    icdf = fold->get(snam::ICDF); 
}

inline void sstandard::save(const char* base, const char* rel )
{
    NPFold* fold = serialize(); 
    fold->save(base, rel); 
}

inline void sstandard::load(const char* base, const char* rel )
{
    NPFold* fold = NPFold::Load(base, rel) ; 
    import(fold) ; 
}
 

/**
sstandard::make_bd
-------------------

Create array of shape (num_bd, 4) holding int "pointers" to (omat,osur,isur,imat)

**/

inline NP* sstandard::make_bd( const std::vector<int4>& vbd, const std::vector<std::string>& bdname )
{
    NP* a_bd = NPX::ArrayFromVec<int, int4>( vbd );  
    a_bd->set_names( bdname );
    return a_bd ; 
}

/**
sstandard::make_optical
-------------------------

The optical buffer int4 payload has entries for both materials and surfaces. 
Array shape at creation and as persisted::

                  int4 payload
                  |
     (num_bnd, 4, 4)
               |
             (omat,osur,isur,imat)             

Shape at point of use on GPU combines the first two dimensions to give "line" 
access to materials and surfaces::

     (num_bnd*4, 4 ) 

The optical buffer int4 payloads for materials and surfaces:

+------------------+---------------+-------------------------------+--------------------------+-------------------+   
|                  | .x            |  .y   Payload_Y               |  .z                      |  .w               |
+==================+===============+===============================+==========================+===================+
| MATERIAL LINES   |   idx+1       |  UNUSED                       | UNUSED                   |  UNUSED           |
+------------------+---------------+-------------------------------+--------------------------+-------------------+
| SURFACE LINES    |               | type                          | finish                   |  value_percent    |
| [FORMERLY]       |   idx+1       | 0:dielectric_metal            | 0:polished               |                   |
|                  |               | 1:dielectric_dielectric       | 1:polishedfrontpainted   |                   |
|                  |               |                               | 3:ground                 |                   | 
|                  |               +-------------------------------+--------------------------+-------------------+
|                  |               |  YZW : THESE THREE COLUMNS WERE FORMERLY NEVER USED ON DEVICE                |
+------------------+---------------+-------------------------------+--------------------------+-------------------+
| SURFACE LINES    |   idx+1       | smatsur::TypeFromChar(OSN0)   | ZW : AS ABOVE BUT STILL NOT USED ON DEVICE   |      
| [NOW]            |               | [MAT/SUR TYPE ENUM "ems"]     |                                              |
+------------------+---------------+-------------------------------+----------------------------------------------+

Q: How come the yzw columns not used on device ?
A: Because that info is used on CPU to prepare the surface entries 
   of the bnd array, which are accessed on device via the boundary texture. 


HANDLING SIGMA_ALPHA/POLISH GROUND SURFACES ?
-----------------------------------------------

This loops over all surfaces in use in the geometry, so 
can detect surfaces that need special handling : and communicate 
that via the ems smatsur.h enum value.  

**/

inline NP* sstandard::make_optical(
     const std::vector<int4>& vbd, 
     const std::vector<std::string>& suname, 
     const NPFold* surface )
{
    int ni = vbd.size() ; 
    int nj = 4 ; 
    int nk = 4 ; 

    NP* op = NP::Make<int>(ni, nj, nk); 
    int* op_v = op->values<int>(); 

    for(int i=0 ; i < ni ; i++)       // over vbd 
    {
        const int4& bd_ = vbd[i] ; 
        for(int j=0 ; j < nj ; j++)   // over (omat,osur,isur,imat)
        {
            int op_index = i*nj*nk + j*nk ;

            int idx = -2 ; 
            switch(j)
            {
                case 0: idx = bd_.x ; break ;   
                case 1: idx = bd_.y ; break ;   
                case 2: idx = bd_.z ; break ;   
                case 3: idx = bd_.w ; break ;   
            }
            int idx1 = idx+1 ; 
            bool is_mat = j == 0 || j == 3 ; 
            bool is_sur = j == 1 || j == 2 ; 

            if(is_mat)
            {
                assert( idx > -1 );   // omat,imat must always be present
                op_v[op_index+0] = idx1 ; 
                op_v[op_index+1] = 0 ; 
                op_v[op_index+2] = 0 ; 
                op_v[op_index+3] = 0 ; 
            }
            else if(is_sur)
            {
                const char* sn = snam::get(suname, idx) ; 
                if(idx > -1 ) assert(sn) ;  
                // all surf should have name, do not always have surf

                NPFold* surf = sn ? surface->get_subfold(sn) : nullptr ;
                bool is_implicit = sn && strncmp(sn, IMPLICIT_PREFIX, strlen(IMPLICIT_PREFIX) ) == 0 ; 
                int Type = -2 ; 
                int Finish = -2 ; 
                int ModelValuePercent = -2 ; 
                std::string OSN = "-" ; 

                if( is_implicit )
                {
                    assert( surf == nullptr ) ;  // not expecting to find surf for implicits 
                    Type = 1 ; 
                    Finish = 1 ; 
                    ModelValuePercent = 100 ;  // placeholders to match old_optical ones
                    OSN = "X" ;  // Implicits classified as ordinary Surface as they have bnd/sur entries 
                }
                else
                {
                    int missing = 0 ;  // -2 better, but use 0 to match old_optical 
                    Type              = surf ? surf->get_meta<int>("Type",-1) : missing ;
                    Finish            = surf ? surf->get_meta<int>("Finish", -1 ) : missing ;
                    ModelValuePercent = surf ? int(100.*surf->get_meta<double>("ModelValue", 0.)) : missing ; 
                    OSN = surf ? surf->get_meta<std::string>("OpticalSurfaceName", "-") : "-" ; 
                }


                char OSN0 = *OSN.c_str() ;                     
                int ems = smatsur::TypeFromChar(OSN0) ; 

                /**
                HERE CAN DETECT FINISH AND ModelValuePercent THAT
                REQUIRES SIGMA_ALPHA OR POLISH GROUND SURFACE HANDLING   
                FOR WHICH WILL NEED NEW smatsur.h enum value
                **/

                int Payload_Y = ems ; 

                if(VERBOSE) std::cout 
                    << " bnd:i "   << std::setw(3) << i 
                    << " sur:idx " << std::setw(3) << idx 
                    << " Type " << std::setw(2) << Type
                    << " Finish " << std::setw(2) << Finish
                    << " MVP " << std::setw(3) << ModelValuePercent
                    << " surf " << ( surf ? "YES" : "NO " )
                    << " impl " << ( is_implicit ? "YES" : "NO " )
                    << " osn0 " << ( OSN0 == '\0' ? '0' : OSN0 ) 
                    << " OSN " << OSN 
                    << " ems " << ems
                    << " emsn " << smatsur::Name(ems) 
                    << " sn " << ( sn ? sn : "-" ) 
                    << std::endl 
                    ; 

                op_v[op_index+0] = idx1 ; 
                op_v[op_index+1] = Payload_Y ; 
                op_v[op_index+2] = Finish ; 
                op_v[op_index+3] = ModelValuePercent ; 
            }
        }
    }
    return op ; 
}




/**
sstandard::make_bnd
---------------------

Form bnd array by interleaving mat and sur array entries as directed by vbd int pointers. 

**/

inline NP* sstandard::make_bnd( 
    const std::vector<int4>& vbd, 
    const std::vector<std::string>& bdname, 
    const NP* mat, 
    const NP* sur )
{
    assert( mat->shape.size() == 4 ); 
    assert( sur->shape.size() == 4 ); 

    int num_mat = mat->shape[0] ; 
    int num_sur = sur->shape[0] ; 

    for(int d=1 ; d < 4 ; d++) assert( mat->shape[d] == sur->shape[d] ) ; 

    assert( mat->shape[1] == sprop::NUM_PAYLOAD_GRP ); 
    int num_domain = mat->shape[2] ; 
    assert( mat->shape[3] == sprop::NUM_PAYLOAD_VAL ); 

    const double* mat_v = mat->cvalues<double>(); 
    const double* sur_v = sur->cvalues<double>(); 

    int num_bnd = vbd.size() ; 
    int num_bdname = bdname.size() ; 
    assert( num_bnd == num_bdname );  

    int4 mn ; 
    int4 mx ;
    column_range(mn, mx, vbd ); 
    if(VERBOSE) std::cout << " sstandard::bnd mn " << mn << " mx " << mx << std::endl ;  

    assert( mx.x < num_mat ); 
    assert( mx.y < num_sur ); 
    assert( mx.z < num_sur ); 
    assert( mx.w < num_mat ); 

    int ni = num_bnd ;                // ~53                 
    int nj = sprop::NUM_MATSUR ;      //   4  (omat,osur,isur,imat)
    int nk = sprop::NUM_PAYLOAD_GRP ; //   2
    int nl = num_domain ;             // 761  fine domain
    int nn = sprop::NUM_PAYLOAD_VAL ; //   4

    int np = nk*nl*nn ;               // 2*761*4  number of payload values for one mat/sur 


    NP* bnd_ = NP::Make<double>(ni, nj, nk, nl, nn ); 
    bnd_->fill<double>(-1.) ; // trying to match X4/GGeo unfilled 
    bnd_->set_names( bdname ); 

    // metadata needed by QBnd::MakeBoundaryTex
    bnd_->set_meta<float>("domain_low",  sdomain::DomainLow() ); 
    bnd_->set_meta<float>("domain_high", sdomain::DomainHigh() ); 
    bnd_->set_meta<float>("domain_step", sdomain::DomainStep() ); 
    bnd_->set_meta<float>("domain_range", sdomain::DomainRange() ); 

    double* bnd_v = bnd_->values<double>() ; 

    for(int i=0 ; i < ni ; i++)
    {
        std::array<int, 4> _bd = {{ vbd[i].x, vbd[i].y, vbd[i].z, vbd[i].w }} ; 
        for(int j=0 ; j < nj ; j++)
        {
            int ptr     = _bd[j] ;  // omat,osur,isur,imat index "pointer" into mat or sur arrays
            if( ptr < 0 ) continue ; 
            bool is_mat =  j == 0 || j == 3 ; 
            bool is_sur =  j == 1 || j == 2 ; 
            if(is_mat) assert( ptr < num_mat ); 
            if(is_sur) assert( ptr < num_sur ); 

            int src_index = ptr*np ; 
            int dst_index = (i*nj + j)*np ; 
            const double* src_v = is_mat ? mat_v : sur_v ; 

            for(int p=0 ; p < np ; p++) bnd_v[dst_index + p] = src_v[src_index + p] ; 
        }
    }
    return bnd_  ; 
}

inline void sstandard::column_range(int4& mn, int4& mx,  const std::vector<int4>& vbd)
{
    mn.x = std::numeric_limits<int>::max() ; 
    mn.y = std::numeric_limits<int>::max() ; 
    mn.z = std::numeric_limits<int>::max() ; 
    mn.w = std::numeric_limits<int>::max() ; 

    mx.x = std::numeric_limits<int>::min() ; 
    mx.y = std::numeric_limits<int>::min() ; 
    mx.z = std::numeric_limits<int>::min() ; 
    mx.w = std::numeric_limits<int>::min() ; 

    int num = vbd.size(); 
    for(int i=0 ; i < num ; i++)
    {
        const int4& b = vbd[i] ; 
        if(b.x > mx.x) mx.x = b.x ; 
        if(b.y > mx.y) mx.y = b.y ; 
        if(b.z > mx.z) mx.z = b.z ; 
        if(b.w > mx.w) mx.w = b.w ;

        if(b.x < mn.x) mn.x = b.x ; 
        if(b.y < mn.y) mn.y = b.y ; 
        if(b.z < mn.z) mn.z = b.z ; 
        if(b.w < mn.w) mn.w = b.w ;
    }
}

/**
sstandard::unused_mat
-------------------------

This now done at U4Tree.h level with U4Material::MakeStandardArray
to allow use of Geant4 interpolation. 

This operates from the NPFold props using NP interpolation. 
In principal it should give equivalent results to Geant4 interpolation. 
However its simpler to just use Geant4 interpolation from U4Tree level. 

**/
inline NP* sstandard::unused_mat( const std::vector<std::string>& names, const NPFold* fold )
{
    assert(0); 
    const sproplist* pl = sproplist::Material() ; 
    return unused_create(pl, names, fold ); 
}

/**
sstandard::unused_sur
-----------------------

This is now done with U4Tree::initSurfaces_Serialize using U4SurfaceArray

Note that because the sur array is not a one-to-one from properties
like the mat array this approach is anyhow unworkable as it stands. 

**/

inline NP* sstandard::unused_sur( const std::vector<std::string>& names, const NPFold* fold )
{
    assert(0); 
    const sproplist* pl = sproplist::Surface() ; 
    return unused_create(pl, names, fold ); 
}

/**
sstandard::unused_create
--------------------------

This assumes simple one-to-one relationship between the props 
and the array content. That is true for "mat" but not for "sur"

**/

inline NP* sstandard::unused_create(const sproplist* pl, const std::vector<std::string>& names, const NPFold* fold )
{ 
    assert(0); 
    sdomain dom ; 
     
    int ni = names.size() ;
    int nj = sprop::NUM_PAYLOAD_GRP ; 
    int nk = dom.length ; 
    int nl = sprop::NUM_PAYLOAD_VAL ; 

    NP* sta = NP::Make<double>(ni, nj, nk, nl) ; 
    sta->set_names(names); 
    double* sta_v = sta->values<double>(); 

    std::cout << "sstandard::create sta.sstr " << sta->sstr() << std::endl ; 

    for(int i=0 ; i < ni ; i++ )               // names
    {
        const char* name = names[i].c_str() ; 
        NPFold* sub = fold->get_subfold(name) ; 

        std::cout 
            << std::setw(4) << i 
            << " : "
            << std::setw(60) << name
            << " : "
            << sub->stats()
            << std::endl 
            ;

        for(int j=0 ; j < nj ; j++)           // payload groups
        {
            for(int k=0 ; k < nk ; k++)       // wavelength 
            {
                //double wavelength_nm = dom.wavelength_nm[k] ; 
                double energy_eV = dom.energy_eV[k] ; 
                double energy = energy_eV * 1.e-6 ;  // Geant4 actual energy unit is MeV

                for(int l=0 ; l < nl ; l++)   // payload values
                {
                    const sprop* prop = pl->get(j,l) ; 
                    assert( prop ); 

                    const char* pn = prop->name ; 
                    const NP* a = sub->get(pn) ; 
                    double value = a ? a->interp( energy ) : prop->def ; 

                    int index = i*nj*nk*nl + j*nk*nl + k*nl + l ; 
                    sta_v[index] = value ; 
                }
            }
        }
    }
    return sta ; 
}


