
#include <iostream>
#include <boost/lexical_cast.hpp>




void check(const char* value)
{
     float percent = boost::lexical_cast<float>(value)*100.f ;   // express as integer percentage 
 
     std::cout << " value "  << value 
               << " percent " << percent
               << std::endl ; 

     unsigned upercent = percent ;
      
     std::cout << " value "  << value 
               << " percent " << percent
               << " upercent " << upercent
               << std::endl ; 
}





int main()
{
     check("0.99");
     check("0.999");

     return 0 ; 
}

