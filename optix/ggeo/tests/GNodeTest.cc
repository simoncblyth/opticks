#include <cassert>
#include "GNode.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;


    GNode* node = new GNode(0, NULL, NULL );
    assert(node->getIndex() == 0 );



    return 0 ;
}

