// op --flags 

#include "Opticks.hh"
//#include "GCache.hh"
#include "OpticksFlags.hh"
#include "OpticksAttrSeq.hh"
#include "Index.hpp"

int main(int argc, char** argv)
{
    Opticks ok(argc, argv) ; 

    //GCache gc(&ok);

    OpticksFlags gf(&ok);

    OpticksAttrSeq* q = gf.getAttrIndex(); 

    q->dump();

    Index* idx = gf.getIndex();

    idx->setExt(".ini");

    idx->save(ok.getIdPath());



    return 0 ; 
}
