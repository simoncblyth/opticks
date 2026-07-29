#include "QPlanck.hh"
#include "QPlanck_test.h"

int main()
{
    QPlanck planck ;
    std::cout << planck.desc() << "\n" ;

    size_t num_values = 1'000'000 ;
    NP* wavelength = NP::Make<float>( num_values );
    unsigned long seed = 42 ;
    qplanck_test<float>(wavelength->values<float>(), num_values, seed, planck.d_planck);

    wavelength->save("$FOLD/wavelength.npy");

    std::cout << " wavelength " << ( wavelength ? wavelength->sstr() : "-" ) << "\n" ;


    return 0;
}
