// op --flags 

#include "Opticks.hh"
#include "OpticksFlags.hh"
#include "OpticksAttrSeq.hh"
#include "Index.hpp"

int main(int argc, char** argv)
{
    Opticks ok(argc, argv) ; 

    //OpticksFlags gf(&ok);

    OpticksAttrSeq* q = ok.getFlagNames(); 

    q->dump();

    //OpticksFlags gf();
    //Index* idx = gf.getIndex();
    //idx->setExt(".ini");
    //idx->save(ok.getIdPath());



    return 0 ; 
}
