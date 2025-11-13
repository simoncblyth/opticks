#include <iostream>
#include <vector>
#include <csignal>

#include "NP.hh"
#include "ssys.h"
#include "scuda.h"
#include "squad.h"
#include "sphotonlite.h"
#include "OpticksPhoton.h"

#include "SU.hh"
#include "SPM.hh"

struct SPM_test
{
    NP*    _photonlite ;
    size_t num_photonlite ;
    sphotonlite* photonlite ;
    sphotonlite* d_photonlite ;

    float time_window_ns ;
    unsigned select_flagmask ;


    SPM_test();
    void init();

    int hitlite();
    int merge_partial_select();
    int merge_incremental();
    int merge_partial_select_merge_incremental();

    int test(int argc, char** argv);
};


inline SPM_test::SPM_test()
    :
    _photonlite(NP::Load("$AFOLD/photonlite.npy")),
    num_photonlite(_photonlite ? _photonlite->shape[0] : 0),
    photonlite( _photonlite ? (sphotonlite*)_photonlite->bytes() : nullptr),
    d_photonlite(photonlite ? SU::upload(photonlite, num_photonlite) : nullptr ),
    time_window_ns(1.0f),
    select_flagmask(EFFICIENCY_COLLECT)
{
    init();
}

inline void SPM_test::init()
{
    std::cout << " photonlite " << ( _photonlite ? _photonlite->sstr() : "-" ) << "\n" ;
}

inline int SPM_test::test(int argc, char** argv)
{
    const char* test = "merge_partial_select" ;
    const char* TEST = ssys::getenvvar("TEST", test );
    bool ALL = 0 == strcmp(TEST, "ALL") ;

    std::cout << argv[0] << " TEST[" << TEST << "]\n" ;

    int rc = 0 ;
    if(ALL||0==strcmp(TEST,"merge_partial_select")) rc += merge_partial_select() ;
    if(ALL||0==strcmp(TEST,"merge_incremental"))    rc += merge_incremental() ;
    if(ALL||0==strcmp(TEST,"merge_partial_select_merge_incremental")) rc += merge_partial_select_merge_incremental() ;

    return rc ;
}

int SPM_test::merge_partial_select()
{
    cudaStream_t stream = 0 ;
    sphotonlite* d_mhitlite = nullptr ;
    int          mhitlite_count = 0 ;

    SPM::merge_partial_select(
            d_photonlite,
            num_photonlite,
            &d_mhitlite,
            &mhitlite_count,
            select_flagmask,
            time_window_ns,
            stream );

    std::cout << "SPM_test::merge_partial_select mhitlite_count " << mhitlite_count << "\n" ;

    std::string path = "partial_0.bin" ;
    SPM::save_partial(d_mhitlite, mhitlite_count, path, stream);
    SPM::free(d_mhitlite);

    return 0 ;
}

int SPM_test::merge_incremental()
{
    const char* paths[] = {
        "partial_0.bin",
        "partial_1.bin"
    };

    sphotonlite* d_final = nullptr;
    int          final_n = 0;
    cudaStream_t stream = 0 ;

    SPM::merge_incremental( paths, &d_final, &final_n, time_window_ns, stream );

    std::vector<sphotonlite> h_final(final_n);
    cudaMemcpy(h_final.data(), d_final, final_n*sizeof(sphotonlite), cudaMemcpyDeviceToHost);
    SPM::free(d_final);

    return 0 ;
}


int SPM_test::merge_partial_select_merge_incremental()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    sphotonlite* d_partial = nullptr ;
    int          n_partial = 0 ;

    SPM::merge_partial_select(d_photonlite, num_photonlite, &d_partial, &n_partial,
                              select_flagmask, time_window_ns, stream);

    SPM::save_partial(d_partial, n_partial, "partial_0.bin", stream);


    // TODO : a second one

    // ... later
    const char* paths[] = { "partial_0.bin", "partial_1.bin", nullptr };
    sphotonlite* d_final = nullptr;
    int          n_final = 0;

    SPM::merge_incremental(paths, &d_final, &n_final, time_window_ns, stream);

    // Use d_final...

    cudaStreamSynchronize(stream);

    return 0 ;
}





int main(int argc, char** argv)
{
    SPM_test t ;
    return t.test(argc, argv);
}


