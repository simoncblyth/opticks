#pragma once
/**
QScintThree.h
==============

Prepare a layered texture atlas for the generation of scintillation wavelengths
in the three-species LS model. Only PPO and bisMSB emit photons (LAB is the
solvent, does not emit), so the atlas has 2 emitting species: PPO, bisMSB.

**/

#include <string>
#include "scuda.h"
struct NP ;
template <typename T> struct QTexLayered ;
template <typename T> struct QTex ;
struct qscint_three ;
struct dim3 ;

struct QScintThree
{
    static const QScintThree* INSTANCE ;
    static const QScintThree* Get();

    static QTexLayered<float>* MakeScintTex(const NP* icdf_three );
    static double ConstDouble( const NP* const_three, const char* key );
    static QTex<float4>* MakePropTex(const NP* prop_three, unsigned line );
    static qscint_three* MakeDevInstance(const QTexLayered<float>* icdf_tex, const QTex<float4>* abslength_tex, const QTex<float4>* reemissionprob_tex, const NP* const_three, const NP* prop_three);
    template<typename T> static T* UploadInstance(T* ptr);
    static void ConfigureLaunch( dim3& numBlocks, dim3& threadsPerBlock, size_t width, size_t height );

    QTexLayered<float>* icdf_tex ;
    QTex<float4>*       abslength_tex ;
    QTex<float4>*       reemissionprob_tex ;
    qscint_three*       scint ;
    qscint_three*       d_scint ;

    QScintThree(const NP* icdf_three, const NP* const_three, const NP* prop_three );
    std::string desc() const ;

    NP* wavelength(size_t num_species, size_t num_wavelength, bool hd ) const ;

};

#include "QTexLayered.h"
#include "QTex.hh"
#include "qscint_three.h"

inline QScintThree::QScintThree(const NP* icdf_three, const NP* const_three, const NP* prop_three )
    :
    icdf_tex(MakeScintTex(icdf_three)),
    abslength_tex(MakePropTex(prop_three, 0u)),
    reemissionprob_tex(MakePropTex(prop_three, 1u)),
    scint(MakeDevInstance(icdf_tex, abslength_tex, reemissionprob_tex, const_three, prop_three)),
    d_scint(UploadInstance<qscint_three>(scint))
{
    INSTANCE = this ;
}

inline std::string QScintThree::desc() const
{
    std::stringstream ss ;
    ss << "[QScintThree.desc\n" ;
    ss << "icdf_tex           :" << ( icdf_tex           ? icdf_tex->desc()           : "-" ) << "\n" ;
    ss << "abslength_tex      :" << ( abslength_tex      ? abslength_tex->desc()      : "-" ) << "\n" ;
    ss << "reemissionprob_tex :" << ( reemissionprob_tex ? reemissionprob_tex->desc() : "-" ) << "\n" ;
    ss << "scint    :" << scint << "\n" ;
    if(scint) ss
        << " domain: nm0 " << scint->nm0 << " nms " << scint->nms << " nx " << scint->nx
        << " ls_material_index " << scint->ls_material_index
        << "\n"
        << " species          PPO(0)  BIS(1)\n"
        << " m_tau_intrinsic  " << scint->m_tau_intrinsic[0] << "  " << scint->m_tau_intrinsic[1] << "\n"
        << " m_prob_lab2      " << scint->m_prob_lab2[0]     << "  " << scint->m_prob_lab2[1]     << "\n"
        << " m_tau_lab2       " << scint->m_tau_lab2[0]      << "  " << scint->m_tau_lab2[1]      << "\n"
        ;
    ss << "d_scint  :" << d_scint << "\n" ;
    ss << "]QScintThree.desc\n" ;
    std::string str = ss.str() ;
    return str;
}


/**
QScintThree::MakeScintTex
--------------------------

Convert the icdf_three array of shape (2, 3, 4096, 1) into a 6 layered texture
(2 emitting species PPO/bisMSB x 3 hd zones)

**/

inline QTexLayered<float>* QScintThree::MakeScintTex(const NP* icdf_three )  // static
{
    const NP* src = icdf_three->ebyte == 4 ? icdf_three : NP::MakeNarrow(icdf_three) ;


    bool expected_shape = src->has_shape(2, 3, 4096, 1) ;
    if(!expected_shape) std::cerr << "QScintThree::MakeScintTex unexpected shape of src " << ( src ? src->sstr() : "-" ) << "\n" ;
    assert( expected_shape );
    if(!expected_shape) std::raise(SIGINT);

    bool src_expect = src->uifc == 'f' && src->ebyte == 4 ;
    if(!src_expect) std::cerr << "QScintThree::MakeScintTex unexpected src.uifc{" << src->uifc << "}" << " or src.ebyte{" << src->ebyte << "}\n" ;
    assert( src_expect );
    if(!src_expect) std::raise(SIGINT);

    bool disable_interpolation = ssys::getenvbool("QSCINTTHREE_DISABLE_INTERPOLATION");
    char filterMode = disable_interpolation ? 'P' : 'L' ;
    if(disable_interpolation) std::cerr << "QScintThree::MakeScintTex QSCINTTHREE_DISABLE_INTERPOLATION active using filterMode " << filterMode << "\n" ;


    bool scrunch_height = true ;  // (2, 3, 4096, 1) ->  (6, 1, 4096, 1)  giving 6 layer tex
    QTexLayered<float>* tex = new QTexLayered<float>(src, filterMode, scrunch_height );
    tex->uploadMeta();

    std::cout << "QScintThree::MakeScintTex tex.desc " << ( tex ? tex->desc() : "-" ) << "\n" ;

    return tex ;
}

/**
QScintThree::ConstDouble
-------------------------

Read one of the six LS new-optical-model constants from the const_three array
((N,) double with names, prepared by U4ScintThree::MakeConstArray), then
narrowed to the float 32 bits qscint_three fields.

**/

inline double QScintThree::ConstDouble( const NP* const_three, const char* key ) // static
{
    NP::INT idx = const_three ? const_three->get_name_index(key) : -1 ;

    bool expect = idx >= 0 ;
    if(!expect) std::cerr
        << "QScintThree::ConstDouble missing const_three [" << key << "]"
        << " : re-convert geometry (expect SSim/extra/jnm/const_three.npy)\n" ;
    assert( expect );

    return const_three->cvalues<double>()[idx] ;
}

/**
QScintThree::MakePropTex
-------------------------

Creates a wavelength-domain float4 texture from one line (row) of the
LS-only prop_three array of shape (2, 761, 4) ::

    line 0: ( fraction_lab, fraction_ppo, fraction_bis, abslength_combined )
    line 1: ( reemission_prob_ppo, reemission_prob_bis, spare, spare )

**/

inline QTex<float4>* QScintThree::MakePropTex(const NP* prop, unsigned line )  // static
{
    const NP* src = prop->ebyte == 4 ? prop : NP::MakeNarrow(prop) ;

    bool src_expect = src->uifc == 'f' && src->ebyte == 4 && src->shape.size() == 3 && src->shape[2] == 4 ;
    if(!src_expect) std::cerr << "QScintThree::MakePropTex unexpected src " << ( src ? src->sstr() : "-" ) << "\n" ;
    assert( src_expect );

    [[maybe_unused]] unsigned ny = src->shape[0] ;
    unsigned nx = src->shape[1] ;
    assert( line < ny );

    const float* values = src->cvalues<float>() + size_t(line)*nx*4u ;

    char filterMode = 'L' ;
    bool normalizedCoords = true ;
    QTex<float4>* tex = new QTex<float4>(nx, 1, values, filterMode, normalizedCoords, src) ;

    bool has_meta = src->has_meta() ;
    if(!has_meta) std::cerr << "QScintThree::MakePropTex MISSING domain metadata " << "\n" ;
    assert( has_meta );

    quad domainX ;
    domainX.f.x = src->get_meta<float>("domain_low",   0.f );
    domainX.f.y = src->get_meta<float>("domain_high",  0.f );
    domainX.f.z = src->get_meta<float>("domain_step",  0.f );
    domainX.f.w = src->get_meta<float>("domain_range", 0.f );

    assert( domainX.f.y > domainX.f.x );
    assert( domainX.f.z > 0.f );
    assert( domainX.f.w == domainX.f.y - domainX.f.x );

    tex->setMetaDomainX(&domainX);
    tex->uploadMeta();

    std::cout << "QScintThree::MakePropTex line " << line << " tex.desc " << ( tex ? tex->desc() : "-" ) << "\n" ;

    return tex ;
}

inline qscint_three* QScintThree::MakeDevInstance(
    const QTexLayered<float>* icdf_tex,
    const QTex<float4>* abslength_tex,
    const QTex<float4>* reemissionprob_tex,
    const NP* const_three,
    const NP* prop_three ) // static
{
    qscint_three* scint = new qscint_three ;

    scint->icdf           = icdf_tex->tex ;
    scint->abslength      = abslength_tex->texObj ;
    scint->reemissionprob = reemissionprob_tex->texObj ;

    // domain and LS material index from the prop_three metas (see U4ScintThreeExtra::MakePropArray / U4Tree::initScintThree)
    scint->nm0 = prop_three->get_meta<float>("domain_low",  0.f ) ;
    scint->nms = prop_three->get_meta<float>("domain_step", 0.f ) ;
    scint->nx  = unsigned(prop_three->shape[1]) ;

    int ls_idx = prop_three->get_meta<int>("ls_material_index", -1 ) ;
    bool ls_expect = ls_idx >= 0 ;
    if(!ls_expect) std::cerr << "QScintThree::MakeDevInstance missing ls_material_index meta: re-convert geometry" << "\n" ;
    assert( ls_expect );
    scint->ls_material_index = unsigned(ls_idx) ;

    // six LS constants from the const_three array (8 Bytes with names),
    // narrowed to the float 4 Bytes device struct, loaded via a mapping table
    static const char* keys[3][2] = {          // column order: PPO, BISMSB
        { "PPO_INTRINSICTAU", "BISMSB_INTRINSICTAU" },
        { "LAB2PPO_PROB",     "LAB2BIS_PROB"        },
        { "LAB2PPO_TAU",      "LAB2BIS_TAU"         } } ;
    float* dst[3] = { scint->m_tau_intrinsic, scint->m_prob_lab2, scint->m_tau_lab2 } ;
    for(int g=0 ; g < 3 ; g++) for(unsigned i=0 ; i < 2 ; i++)
        dst[g][i] = float( ConstDouble(const_three, keys[g][i]) ) ;

    return scint ;
}

template<typename T>
inline T* QScintThree::UploadInstance(T* ptr)  // static
{
    size_t size = sizeof(T);
    T* d_ptr = nullptr ;
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_ptr ), size ));
    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( d_ptr ), ptr, size, cudaMemcpyHostToDevice ));
    return d_ptr ;
}


inline void QScintThree::ConfigureLaunch( dim3& numBlocks, dim3& threadsPerBlock, size_t width, size_t height )  // static
{
    threadsPerBlock.x = 512 ;
    threadsPerBlock.y = 1 ;
    threadsPerBlock.z = 1 ;

    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ;
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ;

    std::cerr
        << "QScintThree::ConfigureLaunch"
        << " width " << std::setw(7) << width
        << " height " << std::setw(7) << height
        << " width*height " << std::setw(7) << width*height
        << " threadsPerBlock"
        << "("
        << std::setw(3) << threadsPerBlock.x << " "
        << std::setw(3) << threadsPerBlock.y << " "
        << std::setw(3) << threadsPerBlock.z << " "
        << ")"
        << " numBlocks "
        << "("
        << std::setw(3) << numBlocks.x << " "
        << std::setw(3) << numBlocks.y << " "
        << std::setw(3) << numBlocks.z << " "
        << ")"
        << "\n"
        ;
}


extern "C" void QScintThree_wavelength(  dim3 numBlocks, dim3 threadsPerBlock, qscint_three* scint, float* wavelength, size_t width, size_t height, bool hd );
extern "C" int QScintThree_wavelength_WITH_LERP();



inline NP* QScintThree::wavelength(size_t num_species, size_t num_wavelength, bool hd ) const
{
    size_t height = num_species ;
    size_t width = num_wavelength ;
    size_t size = width*height*sizeof(float) ;

    float* d_wavelength = nullptr ;
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_wavelength ), size ));

    dim3 numBlocks, threadsPerBlock ;
    ConfigureLaunch(numBlocks, threadsPerBlock, width, height);

    QScintThree_wavelength(numBlocks, threadsPerBlock, d_scint, d_wavelength, width, height, hd ) ;

    NP* wl = NP::Make<float>(num_species, num_wavelength) ;
    wl->set_meta<int>("WITH_LERP", QScintThree_wavelength_WITH_LERP() );

    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( wl->bytes() ), d_wavelength , size, cudaMemcpyDeviceToHost ));
    QUDA_CHECK( cudaFree(d_wavelength) );

    return wl ;
}


