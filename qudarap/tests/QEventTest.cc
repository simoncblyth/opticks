#include "OPTICKS_LOG.hh"

#include <cuda_runtime.h>

#include "SRng.hh"
#include "SPath.hh"

#include "scuda.h"
#include "stran.h"
#include "sqat4.h"
#include "ssincos.h"

#include "QBuf.hh"
#include "QEvent.hh"

const char* BASE = "$TMP/qudarap/QEventTest" ; 


const NP* test_MakeCountGensteps()
{
    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };  
    unsigned x_total = 0 ; 
    for(unsigned i=0 ; i < photon_counts_per_genstep.size() ; i++) x_total += photon_counts_per_genstep[i] ; 
    const NP* gs = QEvent::MakeCountGensteps(photon_counts_per_genstep) ; 

    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve(BASE, create_dirs ); 
    gs->save(fold, "cngs.npy"); 

    return gs ; 
}


const NP* test_MakeCenterExtentGensteps(int nx, int ny, int nz)
{
    float4 ce = make_float4( 1.f, 2.f, 3.f, 100.f ); 

    std::vector<int> cegs = {{nx, ny, nz, 10 }} ; 
    float gridscale = 1.f ; 

    //bool rot = false ;  // 45 degress around Z   OR identity 
    //const Tran<float>* tr = rot ? Tran<float>::make_rotate( 0., 0., 1., 45. ) : Tran<float>::make_identity() ;
    //std::cout << " tr " << *tr << std::endl ; 
    //qat4* qt_ptr = new qat4( tr->tdata() ); 


    const char* str = "(-0.585,-0.805, 0.098, 0.000) (-0.809, 0.588, 0.000, 0.000) (-0.057,-0.079,-0.995, 0.000) (1022.116,1406.822,17734.953, 1.000)"  ;
    qat4* qt = qat4::from_string(str); 
    const Tran<double>* geotran = Tran<double>::ConvertToTran(qt); 

     


    const NP* gs = QEvent::MakeCenterExtentGensteps(ce, cegs, gridscale, geotran ); 

    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve(BASE, create_dirs ); 
    gs->save(fold, "cegs.npy"); 

    return gs ; 
}

void test_GenerateCenterExtentGensteps( const NP* gsa )
{
    LOG(info) << " gsa " << gsa->sstr() ; 
    assert( gsa->shape.size() == 3 && gsa->shape[1] == 6 && gsa->shape[2] == 4 ); 
    std::vector<quad6> gsv(gsa->shape[0]) ; 
    memcpy( gsv.data(), gsa->bytes(), gsa->arr_bytes() ); 

    std::vector<quad4> pp ; 
    quad4 p ; 
    p.zero(); 

    unsigned seed = 0 ; 
    SRng<float> rng(seed) ;
 
    for(unsigned i=0 ; i < gsv.size() ; i++)
    {
        const quad6& gs = gsv[i]; 
        qat4 qt(gs) ;  // transform from last 4 quads of genstep 

        unsigned num_photons = gs.q0.u.w ; 
        std::cout << " i " << i << " num_photons " << num_photons << std::endl ; 

        double u, phi, sinPhi, cosPhi ; 

        for(unsigned j=0 ; j < num_photons ; j++)
        {
            //u = rng(); 
            u = double(j)/double(num_photons-1) ; 

            phi = 2.*M_PIf*u ;   
            ssincos(phi,sinPhi,cosPhi);

            p.q0.f = gs.q1.f ; 

            p.q1.f.x = cosPhi ;   // direction
            p.q1.f.y = 0.f    ;   
            p.q1.f.z = sinPhi ;   
            p.q1.f.w = 1.f    ;   // weight

            qt.right_multiply_inplace( p.q0.f, 1.f );   // position 
            qt.right_multiply_inplace( p.q1.f, 0.f );   // direction 

            pp.push_back(p) ;  
        }
    }

    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve(BASE, create_dirs ); 

    NP* ppa = NP::Make<float>( pp.size(), 4, 4 ); 
    memcpy( ppa->bytes(),  (float*)pp.data(), ppa->arr_bytes() );

    std::cout << "ppa " << ppa->sstr() << std::endl ; 
    ppa->save(fold, "ppa.npy"); 
}







void test_QEvent(const NP* gs)
{
    QEvent* event = new QEvent ; 
    event->setGensteps(gs); 

    unsigned num_photons = event->getNumPhotons() ; 
    assert( num_photons > 0); 

    LOG(info) << event->desc() ; 
    event->seed->download_dump("event->seed", 10); 
    event->checkEvt(); 

    cudaDeviceSynchronize(); 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //const NP* gs = test_MakeCountGensteps() ; 
    const NP* gs0 = test_MakeCenterExtentGensteps(3, 0, 3) ; 
    assert( gs0 ); 
    gs0->dump(); 

    test_GenerateCenterExtentGensteps(gs0); 

    //test_QEvent(gs); 
    return 0 ; 
}

