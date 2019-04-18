// TEST=NSolidTest om-t

#include <vector>
#include "OPTICKS_LOG.hh"
#include "NSolid.hpp"
#include "NNode.hpp"
#include "NTreeAnalyse.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv) ; 

    typedef std::vector<int> VI ; 
    VI lvs = { -18, -19,  18, 19, 20 , 21 } ;  

    for( VI::const_iterator it=lvs.begin() ; it != lvs.end() ; it++ )
    { 
        int lv = *it ; 
        nnode* a = NSolid::create(lv); 
        if(!a) continue ;
       
        LOG(fatal) << "LV=" << lv << " label " << ( a->label ? a->label : "-" ) ; 
        LOG(error) << NTreeAnalyse<nnode>::Desc(a) ; 
    }


    return 0 ; 
} 
