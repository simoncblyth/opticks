
#include <cassert>

#include "Opticks.hh"
#include "AssimpImporter.hh"

#include "PLOG.hh"
#include "ASIRAP_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    ASIRAP_LOG__ ;

    Opticks ok(argc, argv);

    ok.configure();

    const char* path = ok.getDAEPath();

    assert(path);

    LOG(info) << " import " << path ; 

    AssimpImporter assimp(path);

    assimp.import();




    return 0 ;
}

