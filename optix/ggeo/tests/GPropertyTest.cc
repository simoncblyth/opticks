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
 
    GProperty<float>* pcdf = GProperty<float>::load("/tmp/reemission_cdf.npy");
    pcdf->Summary("pcdf",20);

    float mx = GProperty<float>::maxdiff( rcdf, pcdf );
    printf("maxdiff (rcdf - pcdf)*1e9 : %10.4f\n", mx*1e9 );


    GAry<float>* domdif = GAry<float>::subtract( rcdf->getDomain(), pcdf->getDomain() );
    domdif->Summary("domdif*1e6",20, 1e6); 

    GAry<float>* valdif = GAry<float>::subtract( rcdf->getValues(), pcdf->getValues() );
    valdif->Summary("valdif*1e6",20, 1e6); 


    return 0 ;
}
