#include "GAry.hh"
#include "GProperty.hh"

int main(int argc, char** argv)
{
   /*
    GAry<float>* dom  = GAry<float>::from_constant(10, 1.f) ; 
    GAry<float>* vals = GAry<float>::from_constant(10, 10.f); 

    GProperty<float>* prop = new GProperty<float>(vals, dom );
    prop->Summary("prop", 1);

    GProperty<float>* cdf = prop->createCDF();
    cdf->Summary("cdf", 1);

    GProperty<float>* rcdf = prop->createReciprocalCDF();
    rcdf->Summary("rcdf", 1);
   */

    GProperty<float>* slow = GProperty<float>::load("/tmp/slowcomponent.npy");
    slow->Summary("slow",20);

    GProperty<float>* rcdf = slow->createReciprocalCDF();
    rcdf->Summary("rcdf", 20);
 


    return 0 ;
}
