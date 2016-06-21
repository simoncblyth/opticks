#include "GProperty.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    int N = 3 ; 

    float* val = new float[N] ;
    float* dom = new float[N] ;
  
    for(int i=0 ; i < N ; i++)
    {
        dom[i] = i*1.f ;   
        val[i] = i*10.f ;   
    }

    GProperty<float>* prop = new GProperty<float>(val,dom,N);
    prop->Summary();


    return 0 ; 
}
