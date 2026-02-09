#include "NP.hh"
#include "scuda.h"
#include "squad.h"
#include "ssys.h"
#include "sfr.h"
#include "SU.hh"

#include "MortonOverlapScan.hh"

struct MortonOverlapScan_test
{
    static constexpr const char* WINDOW = "WINDOW" ;
    static constexpr const char* FOCUS = "FOCUS" ;
    int      window ;
    int      _focus ;

    sfr      fr ;
    int      focus ;
    float    x0, y0, z0, x1, y1, z1 ;
    int      bbrc ;
    float    dx ;
    float    dy ;
    float    dz ;
    float    cx ;
    float    cy ;
    float    cz ;


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
    window(ssys::getenvint(WINDOW,4)),
    _focus(ssys::getenvint(FOCUS,-1)),
    fr(sfr::Load_("$MFOLD/sfr.npy")),
    focus(fr.get_prim()),
    bbrc(fr.write_bb(&x0)),
    dx(x1-x0),
    dy(y1-y0),
    dz(z1-z0),
    cx((x0+x1)/2.f),
    cy((y0+y1)/2.f),
    cz((z0+z1)/2.f),
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
        << " fr.name [" << fr.get_name() << "]\n"
        << " fr.treedir [" << fr.get_treedir() << "]\n"
        << " x0 " << std::setw(10) << std::fixed << std::setprecision(3) << x0
        << " y0 " << std::setw(10) << std::fixed << std::setprecision(3) << y0
        << " z0 " << std::setw(10) << std::fixed << std::setprecision(3) << z0
        << "\n"
        << " x1 " << std::setw(10) << std::fixed << std::setprecision(3) << x1
        << " y1 " << std::setw(10) << std::fixed << std::setprecision(3) << y1
        << " z1 " << std::setw(10) << std::fixed << std::setprecision(3) << z1
        << "\n"
        << " dx " << std::setw(10) << std::fixed << std::setprecision(3) << dx
        << " dy " << std::setw(10) << std::fixed << std::setprecision(3) << dy
        << " dz " << std::setw(10) << std::fixed << std::setprecision(3) << dz
        << "\n"
        << " cx " << std::setw(10) << std::fixed << std::setprecision(3) << cx
        << " cy " << std::setw(10) << std::fixed << std::setprecision(3) << cy
        << " cz " << std::setw(10) << std::fixed << std::setprecision(3) << cz
        << "\n"
        << " window " << window
        << " _focus " << _focus
        << " focus " << focus
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

/**
MortonOverlapScan_test::scan
------------------------------

Note that the overlap array is created and saved
even when there are zero overlaps to ensure that
stale prior overlap arrays are replaced.

**/


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
        focus,
        stream
    );

    _overlap = quad4::zeros(num_overlap) ;
    if(num_overlap > 0)
    {
        overlap = (quad4*)_overlap->bytes();
        SU::copy_device_to_host_presized( overlap, d_overlap, num_overlap );
    }
    _overlap->save("$MFOLD/simtrace_overlap.npy") ;

    return 0;
}


inline int MortonOverlapScan_test::test() // static
{
    MortonOverlapScan_test t ;
    t.scan();
    std::cout << t.desc();
    return 0;
}

int main()
{
    return MortonOverlapScan_test::test();
}

