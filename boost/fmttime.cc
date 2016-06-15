#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>

using namespace boost::gregorian;
using namespace boost::posix_time;

int main(int argc, char **argv) 
{

   const char* fmt = argc > 1  ? argv[1] : "%d-%b-%Y %H:%M:%S" ;

   time_facet* facet = new time_facet(fmt);

   std::stringstream ss ;  

   std::locale loc(ss.getloc(), facet); 

   ss.imbue(loc);

   ss << second_clock::local_time() ;

   std::string s = ss.str();

   std::cout
         << argv[0]
         << " fmt " << fmt
         << " current time " << s 
         << std::endl;


    return 0 ; 

}
