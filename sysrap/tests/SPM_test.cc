#include <iostream>
#include <vector>
#include <csignal>
#include <filesystem>
#include <regex>


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
    static constexpr size_t EDGE = 20 ;
    NP*    _photonlite ;
    size_t num_photonlite ;
    sphotonlite* photonlite ;
    sphotonlite* d_photonlite ;


    NP*    _hitlite ;
    size_t num_hitlite ;
    sphotonlite* hitlite ;


    float merge_window_ns ;
    unsigned select_flagmask ;


    SPM_test();
    void init();

    int merge_partial_select();
    int merge_partial_select_async();

    int dump();
    int dump_hitlite();
    int dump_partial();
    int desc_diff();

    int merge_incremental();
    int merge_partial_select_merge_incremental();

    int test(int argc, char** argv);
};


/**
SPM_test::SPM_test
------------------

Expedient to NOT use SEventConfig here, so just directly get the envvar

**/


inline SPM_test::SPM_test()
    :
    _photonlite(NP::Load("$AFOLD/photonlite.npy")),
    num_photonlite(_photonlite ? _photonlite->shape[0] : 0),
    photonlite( _photonlite ? (sphotonlite*)_photonlite->bytes() : nullptr),
    d_photonlite(photonlite ? SU::upload(photonlite, num_photonlite) : nullptr ),
    _hitlite(NP::Load("$AFOLD/hitlite.npy")),
    num_hitlite(_hitlite ? _hitlite->shape[0] : 0),
    hitlite( _hitlite ? (sphotonlite*)_hitlite->bytes() : nullptr),
    merge_window_ns(ssys::getenvfloat("OPTICKS_MERGE_WINDOW",0.f)),
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
    if(ALL||0==strcmp(TEST,"merge_partial_select_async")) rc += merge_partial_select_async() ;
    if(ALL||0==strcmp(TEST,"dump"))                 rc += dump() ;
    if(ALL||0==strcmp(TEST,"dump_partial"))         rc += dump_partial() ;
    if(ALL||0==strcmp(TEST,"dump_hitlite"))         rc += dump_hitlite() ;
    if(ALL||0==strcmp(TEST,"desc_diff"))            rc += desc_diff() ;
    if(ALL||0==strcmp(TEST,"merge_incremental"))    rc += merge_incremental() ;
    if(ALL||0==strcmp(TEST,"merge_partial_select_merge_incremental")) rc += merge_partial_select_merge_incremental() ;

    return rc ;
}

int SPM_test::merge_partial_select()
{
    cudaStream_t stream = 0 ;
    sphotonlite* d_hitlitemerged = nullptr ;
    size_t       num_hitlitemerged = 0 ;

    SPM::merge_partial_select(
            d_photonlite,
            num_photonlite,
            &d_hitlitemerged,
            &num_hitlitemerged,
            select_flagmask,
            merge_window_ns,
            stream );

    std::cout
        << "SPM_test::merge_partial_select" << "\n"
        << " merge_window_ns " << merge_window_ns << "\n"
        << " num_hitlitemerged " << num_hitlitemerged << "\n"
        << " _hitlite.sstr " << ( _hitlite ? _hitlite->sstr() : "-" ) << "\n"
        ;

    std::string path = "partial_0.bin" ;
    SPM::save_partial(d_hitlitemerged, num_hitlitemerged, path, stream);
    SPM::free(d_hitlitemerged);

    return 0 ;
}


int SPM_test::merge_partial_select_async()
{
    cudaStream_t merge_stream ;
    cudaStreamCreate(&merge_stream);

    SPM_MergeResult result = SPM::merge_partial_select_async(
           d_photonlite,
           num_photonlite,
           select_flagmask,
           merge_window_ns,
           merge_stream );
    // merge_stream does all then ready_event recorded on stream once all done

    cudaStream_t download_stream ;
    cudaStreamCreate(&download_stream);

    // make download_stream wait for merge_stream ready_event
    result.wait(download_stream);

    // Now safe to use result.count and result.ptr from completed so far download_stream
    NP* hitlitemerged = sphotonlite::zeros( result.count );
    cudaMemcpyAsync(hitlitemerged->bytes(), result.ptr, result.count * sizeof(sphotonlite), cudaMemcpyDeviceToHost, download_stream);

    std::cout
        << "SPM_test::merge_partial_select_async"
        << " hitlitemerged " << ( hitlitemerged ? hitlitemerged->sstr() : "-" )
        << " result.count " << result.count
        << "\n"
        ;


    return 0 ;
}




int SPM_test::dump()
{
    dump_hitlite();
    dump_partial();
    desc_diff();
    return 0 ;
}
int SPM_test::dump_hitlite()
{
    std::cout
        << "SPM_test::dump_hitlite" << "\n"
        << " _hitlite.sstr " << ( _hitlite ? _hitlite->sstr() : "-" ) << "\n"
        << " num_hitlite " << num_hitlite << "\n"
        ;

    for(size_t i=0 ; i < num_hitlite ; i++ )
    {
        if( i < EDGE || i > (num_hitlite - EDGE) )
        std::cout << std::setw(9) << i << " : " << hitlite[i].desc() << "\n" ;
    }
    return 0 ;
}
int SPM_test::dump_partial()
{
    std::cout
        << "SPM_test::dump_partial" << "\n"
        ;

    for (const auto& e : std::filesystem::directory_iterator("."))
    {
        if (e.is_regular_file() && std::regex_match(e.path().filename().string(),std::regex(R"(^partial_\d+\.bin$)")))
        {
            std::cout << e.path() << '\n';
            std::vector<sphotonlite> ll ;
            sphotonlite::loadbin(ll, e.path().c_str());
            for(size_t i=0 ; i < ll.size() ; i++)
            {
                if( i < EDGE || i > ll.size() - EDGE )
                std::cout << std::setw(9) << i << " : " << ll[i].desc() << "\n" ;
            }
        }
    }
    return 0 ;
}
int SPM_test::desc_diff()
{
    const char* path0 = "partial_0.bin" ;
    std::vector<sphotonlite> ll ;
    sphotonlite::loadbin(ll, path0);
    assert( ll.size() == num_hitlite );
    std::cout << sphotonlite::desc_diff(hitlite, ll.data(), ll.size() ) << "\n" ;
    return 0 ;
}




int SPM_test::merge_incremental()
{
    const char* paths[] = {
        "partial_0.bin",
        "partial_1.bin"
    };

    sphotonlite* d_final = nullptr;
    size_t       final_n = 0;
    cudaStream_t stream = 0 ;

    SPM::merge_incremental( paths, &d_final, &final_n, merge_window_ns, stream );

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
    size_t       n_partial = 0 ;

    SPM::merge_partial_select(d_photonlite, num_photonlite, &d_partial, &n_partial,
                              select_flagmask, merge_window_ns, stream);

    SPM::save_partial(d_partial, n_partial, "partial_0.bin", stream);


    // TODO : a second one

    // ... later
    const char* paths[] = { "partial_0.bin", "partial_1.bin", nullptr };
    sphotonlite* d_final = nullptr;
    size_t       n_final = 0;

    SPM::merge_incremental(paths, &d_final, &n_final, merge_window_ns, stream);

    // Use d_final...

    cudaStreamSynchronize(stream);

    return 0 ;
}





int main(int argc, char** argv)
{
    SPM_test t ;
    return t.test(argc, argv);
}


