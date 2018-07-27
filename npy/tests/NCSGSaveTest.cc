#include "SSys.hh"

#include "NCSG.hpp"
#include "NNode.hpp"
#include "NSphere.hpp"
#include "NSceneConfig.hpp"

#include "OPTICKS_LOG.hh"



NCSG* make_csg()
{

    nsphere* a = new nsphere(make_sphere( 0.000,0.000,0.000,500.000 )) ; a->label = "a" ;   
    nsphere* b = new nsphere(make_sphere( 0.000,0.000,0.000,100.000 )) ; b->label = "b" ;   
    nintersection* ab = new nintersection(nintersection::make_intersection( a, b )) ; ab->label = "ab" ; a->parent = ab ; b->parent = ab ;  ;   
    
    ab->update_gtransforms();
    ab->verbosity = SSys::getenvint("VERBOSITY", 1) ; 
    ab->dump() ; 

    const char* boundary = "Rock//perfectAbsorbSurface/Vacuum" ;
    ab->set_boundary(boundary); 
    const char* gltfconfig = "" ;  
    const NSceneConfig* config = new NSceneConfig(gltfconfig);
    unsigned soIdx = 0 ; 
    unsigned lvIdx = 0 ; 

    NCSG* csg = NCSG::FromNode(ab, config, soIdx, lvIdx);
    return csg ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    NCSG* csg = make_csg();
    csg->dump();
    csg->dump_surface_points("dsp", 20);

    csg->save("$TMP/NCSGSaveTest") ; 


    return 0 ; 
}





