#include "GAry.hh"
#include "GProperty.hh"
#include "assert.h"



void test_createReciprocalCDF()
{

    // input SLOWCOMPONENT property
    GProperty<float>* slow = GProperty<float>::load("/tmp/slowcomponent.npy");
    slow->Summary("slow",20);

    // python/NumPy derived reemission (reciprocal) CDF : which has been validated against Geant4
    GProperty<float>* pcdf = GProperty<float>::load("/tmp/reemission_cdf.npy");
    pcdf->Summary("pcdf",20);

    // GProperty derived reciprocal CDF
    GProperty<float>* rcdf = slow->createReciprocalCDF();
    rcdf->Summary("rcdf", 20);

    // differences between the python and GProperty derived CDFs are precision only 
    //  maxdiff (rcdf - pcdf)*1e9 :   596.0464
    float mx = GProperty<float>::maxdiff( rcdf, pcdf );
    assert(mx < 1e-6) ; 
    printf("maxdiff (rcdf - pcdf)*1e9 : %10.4f\n", mx*1e9 );

   // dumping the differences in domain and values
    GAry<float>* domdif = GAry<float>::subtract( rcdf->getDomain(), pcdf->getDomain() );
    domdif->Summary("domdif*1e6",20, 1e6); 

    GAry<float>* valdif = GAry<float>::subtract( rcdf->getValues(), pcdf->getValues() );
    valdif->Summary("valdif*1e6",20, 1e6); 

}






void test_sampling()
{


}





int main(int argc, char** argv)
{
    test_createReciprocalCDF();
    test_sampling();
    return 0 ;
}
