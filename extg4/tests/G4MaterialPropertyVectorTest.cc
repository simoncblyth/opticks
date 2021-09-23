
/**
G4MaterialPropertyVectorTest.cc
=================================

*G4MaterialPropertyVector*  changed in 1100:
 
* < 1100 typedef for G4PhysicsOrderedFreeVector
* 1100   typedef for G4PhysicsFreeVector  
* 1100  Ordered methods relocated to G4PhysicsFreeVector and G4PhysicsOrderedFreeVector dropped  

Hence to avoid version branching it is beneficial to use G4MaterialPropertyVector
rather than the underlying G4PhysicsOrderedFreeVector/G4PhysicsFreeVector

**/

#include "NP.hh"
#include "X4MaterialPropertyVector.hh"
#include "G4MaterialPropertyVector.hh"

struct G4MaterialPropertyVectorTest
{
    const NP* arr ; 
    const G4MaterialPropertyVector* vec ; 

    G4MaterialPropertyVectorTest(const NP* arr_) 
        :
        arr(arr_),
        vec(X4MaterialPropertyVector::FromArray(arr))
    {
    }

    double GetEnergyDiff(double value)
    {
         double a = arr->pdomain<double>( value ); 
         double b = const_cast<G4MaterialPropertyVector*>(vec)->GetEnergy(value); 
         double ab = std::abs( a - b ); 

         std::cout 
            << " a " << std::setw(10) << std::fixed << std::setprecision(4) << a 
            << " b " << std::setw(10) << std::fixed << std::setprecision(4) << b
            << " ab " << std::setw(10) << std::fixed << std::setprecision(4) << ab
            << std::endl 
            ;

         return ab  ; 
    }
};


int main(int argc, char** argv)
{
      std::vector<double> src = {
        0.,    0., 
        1.,   10., 
        2.,   20., 
        3.,   30., 
        4.,   40., 
        5.,   50., 
        6.,   60., 
        7.,   70., 
        8.,   80., 
        9.,   90.  
    }; 

    NP* arr = NP::Make<double>(10,2); 
    arr->read(src.data()); 
    arr->dump(); 

    G4MaterialPropertyVectorTest t(arr); 

    NP* vv = NP::Linspace<double>(-10., 100., 111 );     
    for(unsigned i=0 ; i < vv->shape[0] ; i++)
    {
        double v = vv->get<double>(i); 
        double df = t.GetEnergyDiff(v); 
        assert( df == 0. ); 
    }

    return 0 ; 
}
