#include <cassert>
#include "GMesh.hh"


#include "PLOG.hh"
#include "GGEO_LOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;  

    GMesh* mesh = GMesh::load("$TMP/mm");
    if(!mesh) return 0 ;

    assert(mesh);
    mesh->Summary("check mesh loading");
    mesh->dump("mesh dump", 10);

    return 0 ;
}
