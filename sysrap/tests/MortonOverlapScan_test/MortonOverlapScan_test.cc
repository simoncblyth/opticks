#include "NP.hh"
#include "scuda.h"
#include "squad.h"
#include "ssys.h"
#include "SU.hh"

#include "MortonOverlapScan.hh"

struct MortonOverlapScan_test
{
    static constexpr const char* bbox_default = "-25000,-25000,-25000,25000,25000,25000" ;
    static constexpr const char* BBOX = "BBOX" ;
    static constexpr const char* WINDOW = "WINDOW" ;
    std::vector<float>* bbox ;
    float    x0, y0, z0 ;
    float    x1, y1, z1 ;
    int      window ;

    NP*      _simtrace ;
    size_t   num_simtrace ;
    quad4*   simtrace ;
    quad4*   d_simtrace ;
    quad4*   d_overlap ;
    size_t   num_overlap ;
    quad4*   overlap ;
    NP*      _overlap ;

    MortonOverlapScan_test();

    int scan();
    std::string desc() const ;

    static int test();
};

inline MortonOverlapScan_test::MortonOverlapScan_test()
    :
    bbox(ssys::getenv_vec<float>(BBOX, bbox_default, ',')),
    x0(bbox->size() > 0 ? (*bbox)[0] : 0.f ),
    y0(bbox->size() > 1 ? (*bbox)[1] : 0.f ),
    z0(bbox->size() > 2 ? (*bbox)[2] : 0.f ),
    x1(bbox->size() > 3 ? (*bbox)[3] : 0.f ),
    y1(bbox->size() > 4 ? (*bbox)[4] : 0.f ),
    z1(bbox->size() > 5 ? (*bbox)[5] : 0.f ),
    window(ssys::getenvint(WINDOW,4)),
    _simtrace(NP::Load("$MFOLD/simtrace.npy")),
    num_simtrace(_simtrace ? _simtrace->shape[0] : 0),
    simtrace(_simtrace ? (quad4*)_simtrace->bytes() : nullptr),
    d_simtrace( simtrace ? SU::upload(simtrace, num_simtrace) : nullptr ),
    d_overlap( nullptr ),
    num_overlap(0),
    overlap(nullptr),
    _overlap(nullptr)
{
}

inline std::string MortonOverlapScan_test::desc() const
{
    std::stringstream ss ;
    ss
        << "[MortonOverlapScan_test::desc\n"
        << " x0 " << x0
        << " y0 " << y0
        << " z0 " << z0
        << " x1 " << x1
        << " y1 " << y1
        << " z1 " << z1
        << " window " << window
        << " _simtrace " << ( _simtrace ? _simtrace->sstr() : "-" ) << "\n"
        << " num_simtrace " << num_simtrace << "\n"
        << " d_simtrace " << ( d_simtrace ? "YES" : "NO " )  << "\n"
        << " d_overlap  " << ( d_overlap ? "YES" : "NO " )  << "\n"
        << " num_overlap " << num_overlap << "\n"
        << " _overlap " << ( _overlap ? _overlap->sstr() : "-" ) << "\n"
        << "]MortonOverlapScan_test::desc\n"
        ;
    std::string str = ss.str() ;
    return str ;
}

inline int MortonOverlapScan_test::scan()
{
    cudaStream_t stream = 0 ;
    MortonOverlapScan::Scan(
        d_simtrace,
        num_simtrace,
        &d_overlap,
        &num_overlap,
        x0, y0, z0,
        x1, y1, z1,
        window,
        stream
    );

    if(num_overlap > 0)
    {
        _overlap = quad4::zeros(num_overlap) ;
        overlap = (quad4*)_overlap->bytes();
        SU::copy_device_to_host_presized( overlap, d_overlap, num_overlap );
    }

    return 0;
}


inline int MortonOverlapScan_test::test() // static
{
    MortonOverlapScan_test t ;
    t.scan();
    std::cout << t.desc();
    if(t._overlap) t._overlap->save("$MFOLD/simtrace_overlap.npy") ;
    return 0;
}

int main()
{
    return MortonOverlapScan_test::test();
}

