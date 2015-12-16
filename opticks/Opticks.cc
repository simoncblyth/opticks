#include "Opticks.hh"
#include "NLog.hpp"


void Opticks::init()
{
}


void Opticks::configureS(const char* name, std::vector<std::string> values)
{
}

void Opticks::configureI(const char* name, std::vector<int> values)
{
}


void Opticks::configureF(const char* name, std::vector<float> values)
{
     if(values.empty())
     {   
         printf("Opticks::parameter_set %s no values \n", name);
     }   
     else    
     {   
         float vlast = values.back() ;

         printf("Opticks::parameter_set %s : %lu values : ", name, values.size());
         for(size_t i=0 ; i < values.size() ; i++ ) printf("%10.3f ", values[i]);
         printf(" : vlast %10.3f \n", vlast );

         //configure(name, vlast);  
     }   
}
 


