#pragma once
/**
QScintThree.h
==============

Prepare a layered texture atlas for the generation of scintillation wavelengths
in a 3 species liquid scintillator model : LAB, PPO, bisMSB

**/

#include <string>
struct NP ;
template <typename T> struct QTexLayered ;
struct qscint_three ;
struct dim3 ;

struct QScintThree
{
    static QTexLayered<float>* MakeScintTex(const NP* layered_icdf );
    static qscint_three* MakeDevInstance(const QTexLayered<float>* tex);
    template<typename T> static T* UploadInstance(T* ptr);
    static void ConfigureLaunch( dim3& numBlocks, dim3& threadsPerBlock, size_t width, size_t height );

    QTexLayered<float>* tex ;
    qscint_three*       scint ;
    qscint_three*       d_scint ;

    QScintThree(const NP* layered_icdf );
    std::string desc() const ;

    NP* wavelength(size_t num_species, size_t num_wavelength, bool hd ) const ;

};

#include "QTexLayered.h"
#include "qscint_three.h"

inline QScintThree::QScintThree(const NP* layered_icdf )
    :
    tex(MakeScintTex(layered_icdf)),
    scint(MakeDevInstance(tex)),
    d_scint(UploadInstance<qscint_three>(scint))
{
}

inline std::string QScintThree::desc() const
{
    std::stringstream ss ;
    ss << "[QScintThree.desc\n" ;
    ss << "tex.desc :" << ( tex ? tex->desc() : "-" ) << "\n" ;
    ss << "scint    :" << scint << "\n" ;
    ss << "d_scint  :" << d_scint << "\n" ;
    ss << "]QScintThree.desc\n" ;
    std::string str = ss.str() ;
    return str;
}


/**
QScintThree::MakeScintTex
--------------------------

Convert layered_icdf array of shape (3, 3, 4096, 1) into a 9 layered texture

**/

inline QTexLayered<float>* QScintThree::MakeScintTex(const NP* layered_icdf )  // static
{
    const NP* src = layered_icdf->ebyte == 4 ? layered_icdf : NP::MakeNarrow(layered_icdf) ;


    bool expected_shape = src->has_shape(3, 3, 4096, 1) ;
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


    bool scrunch_height = true ;  // (3, 3, 4096, 1) ->  (9, 1, 4096, 1)  giving 9 layer tex
    QTexLayered<float>* tex = new QTexLayered<float>(src, filterMode, scrunch_height );
    tex->uploadMeta();

    std::cout << "QScintThree::MakeScintTex tex.desc " << ( tex ? tex->desc() : "-" ) << "\n" ;

    return tex ;
}

inline qscint_three* QScintThree::MakeDevInstance(const QTexLayered<float>* tex) // static
{
    qscint_three* scint = new qscint_three ;
    scint->scint_tex_layered = tex->tex ;
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


void QScintThree::ConfigureLaunch( dim3& numBlocks, dim3& threadsPerBlock, size_t width, size_t height )  // static
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


