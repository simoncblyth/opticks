#include "sblackbody.h"
#include "NP.hh"

int main()
{
    std::vector<int> temps = {{6500,6000,5500,5000,4500}} ; 
    int ni = temps.size(); 
    int nj = 1400 ; 
    int nk = 2 ; 

    NP* a = NP::Make<double>(ni, nj, nk);  
    double* aa = a->values<double>(); 
    for(int i=0 ; i < ni ; i += 1 )
    {
        int temp = temps[i] ; 
        a->names.push_back(std::to_string(temp)); 
        double temp_k = temp ; 

        for(int j=0 ; j < nj ; j += 1 )
        {
            double nm = double(j+100) ; 
            double psr = sblackbody::planck_spectral_radiance( nm, temp_k ) ;  
            int idx = i*nj*nk + j*nk ; 
            aa[idx + 0] = nm ; 
            aa[idx + 1] = psr ; 
        }
    }
    a->save("$FOLD/psr.npy"); 

    return 0 ; 
}
