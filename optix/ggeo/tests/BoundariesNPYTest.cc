

#include "NPY.hpp"
#include "GLMPrint.hpp"



#include "Opticks.hh"
#include "OpticksAttrSeq.hh"

#include "GBndLib.hh"
#include "BoundariesNPY.hpp"

#include "GGEO_CC.hh"


// ggv --boundaries

int main(int argc, char** argv)
{
    Opticks* opticks = new Opticks(argc, argv);

    GBndLib* blib = GBndLib::load(opticks, true );
    OpticksAttrSeq* qbnd = blib->getAttrNames();
    blib->close();     //  BndLib is dynamic so requires a close before setNames is called setting the sequence for OpticksAttrSeq
    std::map<unsigned int, std::string> nm = qbnd->getNamesMap(OpticksAttrSeq::ONEBASED) ;

    //qbnd->dump();
    
    NPY<float>* dpho = NPY<float>::load("oxtorch", "1", "dayabay");
    if(!dpho) return 0 ;



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
