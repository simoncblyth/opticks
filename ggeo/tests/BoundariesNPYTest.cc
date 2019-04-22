

#include "NPY.hpp"
#include "GLMPrint.hpp"



#include "Opticks.hh"
#include "OpticksAttrSeq.hh"

#include "GBndLib.hh"
#include "BoundariesNPY.hpp"

#include "OPTICKS_LOG.hh"

#include "GGEO_BODY.hh"


// ggv --boundaries

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks* ok = new Opticks(argc, argv);
    ok->configure();

    GBndLib* blib = GBndLib::load(ok, true );
    OpticksAttrSeq* qbnd = blib->getAttrNames();
    blib->close();     //  BndLib is dynamic so requires a close before setNames is called setting the sequence for OpticksAttrSeq
    std::map<unsigned int, std::string> nm = qbnd->getNamesMap(OpticksAttrSeq::ONEBASED) ;

    qbnd->dump();
    
    NPY<float>* dpho = NPY<float>::load("oxtorch", "1", "dayabay");
    if(!dpho) 
    {
        LOG(warning) << " failed to load dpho event " ; 
        return 0 ;
    }


    //dpho->Summary();

    BoundariesNPY* boundaries = new BoundariesNPY(dpho);
    boundaries->setBoundaryNames(nm); 
    boundaries->setTypes(NULL);
    boundaries->indexBoundaries();
    boundaries->dump();

    //glm::ivec4 sel = boundaries->getSelection() ;
    //print(sel, "boundaries selection");

    return 0 ; 

}
