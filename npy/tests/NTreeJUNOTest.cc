// TEST=NTreeJUNOTest om-t

#include <vector>
#include "NTreeJUNO.hpp"
#include "NNode.hpp"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const NTreeJUNO::VI& v = NTreeJUNO::LVS ; 

    for( NTreeJUNO::VI::const_iterator it=v.begin() ; it != v.end() ; it++ )
    { 
        int lv = *it ; 
        nnode* a = NTreeJUNO::create(lv);   // -ve lv rationalize
        assert( a && a->label ); 
       
        LOG(fatal) << "LV=" << lv << " label " << a->label ;
        LOG(error) << a->ana_desc() ; 
    }

    return 0 ; 
}



