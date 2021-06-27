#include "G4MaterialPropertyVector.hh"
#include "X4MaterialPropertyVector.hh"
#include "NPY.hpp"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    size_t len = 10 ;  
    G4double* energy = new G4double[len] ;
    G4double* value = new G4double[len] ;
 
    for(int i=0 ; i < int(len) ; i++)
    {
        energy[i] = G4double(i)  ; 
        value[i] = 100.*G4double(i); 
    }

    G4MaterialPropertyVector* vec = new G4MaterialPropertyVector(energy, value, len);  
    X4MaterialPropertyVector* xvec = new X4MaterialPropertyVector(vec); 

    NPY<double>* d = xvec->convert<double>(); 
    d->dump(); 

    NPY<float>*  f = xvec->convert<float>(); 
    f->dump(); 
    
    return 0 ; 
}
