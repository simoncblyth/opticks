// op --flags 

#include "Opticks.hh"
#include "OpticksFlags.hh"
#include "OpticksAttrSeq.hh"
#include "Index.hpp"

#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv) ; 
    ok.configure();

    //OpticksFlags gf(&ok);

    OpticksAttrSeq* q = ok.getFlagNames(); 
    assert( q );

    q->dump();

    //OpticksFlags gf();
    //Index* idx = gf.getIndex();
    //idx->setExt(".ini");
    //idx->save(ok.getIdPath());



    return 0 ; 
}
