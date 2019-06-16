// TEST=GPartsTest om-t

#include <cassert>
#include <vector>

#include "NCSG.hpp"
#include "NPYBase.hpp"
#include "NPYList.hpp"
#include "NNode.hpp"
#include "NNodeSample.hpp"

#include "Opticks.hh"

#include "GParts.hh"
#include "GBndLib.hh"

#include "OPTICKS_LOG.hh"


void test_Adopt()
{
    typedef std::vector<nnode*> VN ;
    VN nodes ; 
    NNodeSample::Tests(nodes);
    
    const char* spec = "Rock//perfectAbsorbSurface/Vacuum" ;

    unsigned verbosity = 2 ; 

    for(unsigned i=0 ; i < nodes.size() ; i++)
    {   
        nnode* n = nodes[i] ; 
        OpticksCSG_t type = n->type ; 
        const char* name = n->csgname();
        assert( type < CSG_UNDEFINED && type > 0 && name != NULL ) ; 

        LOG(info) << "GPartsTest " 
                  << " i " << std::setw(3) << i 
                  << " type " << type
                  << " name " << name
                  ;

        n->set_boundary(spec) ; 

        NCSG* csg = NCSG::Adopt( n );
        csg->setVerbosity(verbosity);

        unsigned ndIdx = i ; 
        GParts* pts = GParts::Make( csg , spec, ndIdx  ) ; 
        pts->dump("GPartsTest");

    }
}


void test_save_empty(GBndLib* bndlib)
{
    GParts pts ; 
    pts.setBndLib(bndlib);
    pts.save("$TMP/GPartsTest_test_save_empty");
}



void test_save_load(GBndLib* bndlib)
{
    const NSceneConfig* config = NULL ; 
    const char* spec = "Rock//perfectAbsorbSurface/Vacuum" ;

    typedef std::vector<nnode*> VN ;
    VN nodes ; 
    NNodeSample::Tests(nodes);

    nnode* n = nodes[0] ;  
    n->set_boundary(spec) ; 

    unsigned soIdx = 0 ; 
    unsigned lvIdx = 0 ; 
    unsigned ndIdx = 0 ; 

    NCSG* csg = NCSG::Adopt( n, config, soIdx, lvIdx );

    csg->setVerbosity(2);  
    GParts* pts = GParts::Make(csg, spec, ndIdx ) ; 
    pts->dump("pts");

    const char* dir = "$TMP/GPartsTest_test_save" ;
    pts->setBndLib(bndlib);

    // saving, registers boundaries : so the mlib and slib must be closed


    pts->save(dir);  // asserts in here for lack of bndlib


    GParts* pts2 = GParts::Load(dir);

    pts2->dump("pts2");
}

void test_load_ncsg_make()
{
     NCSG* csg = NCSG::Load("$TMP/tboolean-box--/0");
     if(!csg) return ; 

     const char* spec = "Rock//perfectAbsorbSurface/Vacuum" ; 

     NPYList* npy = csg->getNPYList(); 
     LOG(info) << " npy " << npy->desc() ; 

     csg->setVerbosity(3); 

     unsigned ndIdx = 0 ; 

     GParts* pts = GParts::Make( csg, spec, ndIdx );
     assert(pts); 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

/*
    Opticks ok(argc, argv);

    bool constituents = true ; 
    GBndLib* bndlib = GBndLib::load(&ok, constituents);
    bndlib->closeConstituents();

    test_save_empty(bndlib);
    test_save_load(bndlib);
*/

    test_load_ncsg_make();

    return 0 ;
}

