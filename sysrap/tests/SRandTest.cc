#include <map>
#include "OPTICKS_LOG.hh"
#include "SRand.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
  
    unsigned num_cat = 10 ; 
    unsigned num_throw = 1000000 ; 

    LOG(info) 
       << " num_cat " << num_cat 
       << " num_throw " << num_throw 
       ;

    std::map<unsigned, unsigned> category_count  ; 

    for(unsigned i=0 ; i < num_throw ; i++) 
    {
        unsigned category = SRand::pick_random_category(num_cat); 
        category_count[category] += 1 ; 
    }

    typedef std::map<unsigned, unsigned>::const_iterator IT ; 
    for(IT it=category_count.begin() ; it != category_count.end() ; it++)
    {
        std::cout 
            << std::setw(10) << it->first 
            << " : "
            << std::setw(10) << it->second
            << std::endl
            ;  
    }
   
    return 0 ; 
}   
