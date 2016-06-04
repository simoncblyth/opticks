#include "Opticks.hh"

//#include "GCache.hh"
#include "GBndLib.hh"
#include "OpticksAttrSeq.hh"


#include "BoundariesNPY.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"

// ggv --boundaries

int main(int argc, char** argv)
{
    Opticks* opticks = new Opticks(argc, argv);
    //GCache* cache = new GCache(opticks);

    GBndLib* blib = GBndLib::load(opticks, true );
    OpticksAttrSeq* qbnd = blib->getAttrNames();
    blib->close();     //  BndLib is dynamic so requires a close before setNames is called setting the sequence for OpticksAttrSeq
    std::map<unsigned int, std::string> nm = qbnd->getNamesMap(OpticksAttrSeq::ONEBASED) ;

    //qbnd->dump();
    
    NPY<float>* dpho = NPY<float>::load("oxtorch", "1", "dayabay");
    //dpho->Summary();

    BoundariesNPY* boundaries = new BoundariesNPY(dpho);
    boundaries->setBoundaryNames(nm); 
    boundaries->setTypes(NULL);
    boundaries->indexBoundaries();
    boundaries->dump();

    //glm::ivec4 sel = boundaries->getSelection() ;
    //print(sel, "boundaries selection");


}
