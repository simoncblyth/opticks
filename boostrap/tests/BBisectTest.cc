#include <iomanip>
#include <iostream>
#include <functional>
#include <boost/math/tools/roots.hpp>


template <class T>
struct fn
{
    T operator()(const T& x )
    {
        return (x-2.)*(x-5.) ; 
    }
};

template <class T>
struct tolerance
{
    bool operator()(const T& min, const T& max )
    {
        return (max - min) < 0.001 ;   
    }
};




int main()
{

   fn<float> f ; 
   tolerance<float> tol ; 

   float min = 1 ; 
   float max = 3 ; 

   std::pair<float, float> r = boost::math::tools::bisect(f, min, max, tol );

   std::cout 
      << " r " << std::setw(15) << std::fixed << std::setprecision(4) << r.first
      << " " << r.second 
      << std::endl 
      ;


    return 0 ; 
}
