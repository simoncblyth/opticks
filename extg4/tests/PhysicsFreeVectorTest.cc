#include "NP.hh"

struct PhysicsFreeVector
{
    PhysicsFreeVector(const NP* a) ; 

    double GetEnergy(double aValue);
    double GetMinValue(){ return dataVector.front(); }
    double GetMaxValue(){ return dataVector.back();  }

    std::size_t FindValueBinLocation(double aValue); 
    std::size_t FindValueBinLocation_(double aValue); 
    double      LinearInterpolationOfEnergy(double aValue, std::size_t bin); 

    size_t numberOfNodes;
    double edgeMin ; 
    double edgeMax ; 
    std::vector<double> dataVector ; 
    std::vector<double> binVector ; 
}; 


PhysicsFreeVector::PhysicsFreeVector(const NP* a )
{
    assert( a->shape.size() == 2 && a->shape[1] == 2 ); 

    for(unsigned i=0 ; i < unsigned(a->shape[0]) ; i++)
    {
        binVector.push_back( a->get<double>(i, 0) ); 
        dataVector.push_back( a->get<double>(i, 1) ); 
    }
    numberOfNodes = a->shape[0] ; 

    edgeMin = binVector.front();
    edgeMax = binVector.back(); 
}



double PhysicsFreeVector::GetEnergy(double aValue)
{
  double e;
  if(aValue <= GetMinValue())
  {
    e = edgeMin;
  }
  else if(aValue >= GetMaxValue())
  {
    e = edgeMax;
  }
  else
  {
    std::size_t closestBin = FindValueBinLocation(aValue);
    e                      = LinearInterpolationOfEnergy(aValue, closestBin);
  }
  return e;
}


/**
PhysicsFreeVector::FindValueBinLocation_
------------------------------------------

For aValue <= dataVector[0] this returns a very large number (unsigned -ve)::

    meta:[]
     i     0 v     -10.00 b_ 18446744073709551615 b          8
     i     1 v      -9.00 b_ 18446744073709551615 b          8
     i     2 v      -8.00 b_ 18446744073709551615 b          8
     i     3 v      -7.00 b_ 18446744073709551615 b          8
     i     4 v      -6.00 b_ 18446744073709551615 b          8
     i     5 v      -5.00 b_ 18446744073709551615 b          8
     i     6 v      -4.00 b_ 18446744073709551615 b          8
     i     7 v      -3.00 b_ 18446744073709551615 b          8
     i     8 v      -2.00 b_ 18446744073709551615 b          8
     i     9 v      -1.00 b_ 18446744073709551615 b          8
     i    10 v       0.00 b_ 18446744073709551615 b          8
     i    11 v       1.00 b_          0 b          0
     i    12 v       2.00 b_          0 b          0
     i    13 v       3.00 b_          0 b          0
     ..
     i    96 v      86.00 b_          8 b          8
     i    97 v      87.00 b_          8 b          8
     i    98 v      88.00 b_          8 b          8
     i    99 v      89.00 b_          8 b          8
     i   100 v      90.00 b_          8 b          8

For aValue > dataVector[-1] this returns numberOfNodes - 1::

     i   101 v      91.00 b_          9 b          8
     i   102 v      92.00 b_          9 b          8
     i   103 v      93.00 b_          9 b          8
     i   104 v      94.00 b_          9 b          8
     i   105 v      95.00 b_          9 b          8
     i   106 v      96.00 b_          9 b          8
     i   107 v      97.00 b_          9 b          8
     i   108 v      98.00 b_          9 b          8
     i   109 v      99.00 b_          9 b          8
     i   110 v     100.00 b_          9 b          8
     vec.numberOfNodes 10

**/

std::size_t PhysicsFreeVector::FindValueBinLocation_(double aValue)
{
  // std::lower_bound returns iterator pointing to first data value > aValue,    
  std::size_t bin = std::lower_bound(dataVector.cbegin(), dataVector.cend(), aValue) - dataVector.cbegin() - 1; 
  return bin ; 
} 


/**
PhysicsFreeVector::FindValueBinLocation
------------------------------------------


                             numberOfNodes - 2
                             |   
                             |
     0  1  2  3  4  5  6  7  8  9     
     +--+--+--+--+--+--+--+--+--+      numberOfNodes = 10 

                             ^         values <= minValue and > maxValue  yield this top bin index.  


**/

std::size_t PhysicsFreeVector::FindValueBinLocation(double aValue)
{
  std::size_t bin = FindValueBinLocation_(aValue); 
  bin = std::min(bin, numberOfNodes - 2); 
  return bin;
}

// --------------------------------------------------------------------
double PhysicsFreeVector::LinearInterpolationOfEnergy(double aValue, 
                                                          std::size_t bin)
{
  double res = binVector[bin];
  double del = dataVector[bin + 1] - dataVector[bin];
  if(del > 0.0)
  {
    res += (aValue - dataVector[bin]) * (binVector[bin + 1] - res) / del;
  }
  return res;
}




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

    NP* a = NP::Make<double>(10,2); 
    a->read(src.data()); 
    a->dump(); 

    PhysicsFreeVector vec(a); 

    NP* vv = NP::Linspace<double>(-10., 100., 111 );     
    for(unsigned i=0 ; i < vv->shape[0] ; i++)
    {
        double v = vv->get<double>(i); 

        size_t bin_ = vec.FindValueBinLocation_(v); 
        size_t bin  = vec.FindValueBinLocation(v); 
        double en = vec.GetEnergy(v); 

        std::cout 
            << " i " << std::setw(5) << i 
            << " v " << std::setw(10) << std::fixed << std::setprecision(2) << v
            << " bin_ " << std::setw(10) << bin_
            << " bin " << std::setw(10) << bin
            << " en " << std::setw(10) << std::fixed << std::setprecision(2) << en
            << std::endl
            ; 
    }

    std::cout << " vec.numberOfNodes " <<  vec.numberOfNodes << std::endl ; 
    return 0 ; 
}
