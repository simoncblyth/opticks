
#include "srec.h"
#include "sphoton.h"
#include "sphotonlite.h"

#include "scuda.h"
#include "squad.h"
#include "sevent.h"

int main()
{
    sevent ev = {} ;


    ev.photon = new sphoton ;
    ev.num_photon = 1 ;
    ev.photonlite = new sphotonlite ;
    ev.num_photonlite = 2 ;



    sphoton* p = ev.get_photon_ptr<sphoton>() ;
    assert( p == ev.photon );

    sphotonlite* l = ev.get_photon_ptr<sphotonlite>() ;
    assert( l == ev.photonlite );

    size_t pnum = ev.get_photon_num<sphoton>() ;
    size_t lnum = ev.get_photon_num<sphotonlite>() ;

    assert( pnum == ev.num_photon );
    assert( lnum == ev.num_photonlite );

    std::cout << " pnum " << pnum << " lnum " << lnum << "\n" ;


    ev.hitmerged = nullptr ;
    ev.hitlitemerged = nullptr ;

    {
        sphoton** ptr = ev.get_hitmerged_ptr_ref<sphoton>();
        size_t*   num = ev.get_hitmerged_num_ref<sphoton>();

        sphoton* d0 = new sphoton ;
        size_t n0 = 101 ;

        *ptr = d0 ;
        *num = n0 ;

        assert( ev.hitmerged == d0 );
        assert( ev.num_hitmerged == n0 );
    }


    {
        sphotonlite** ptr = ev.get_hitmerged_ptr_ref<sphotonlite>();
        size_t*      num  = ev.get_hitmerged_num_ref<sphotonlite>();

        sphotonlite* d0 = new sphotonlite ;
        size_t n0 = 202 ;

        *ptr = d0 ;
        *num = n0 ;

        assert( ev.hitlitemerged == d0 );
        assert( ev.num_hitlitemerged == n0 );
    }



    return 0 ;
}
