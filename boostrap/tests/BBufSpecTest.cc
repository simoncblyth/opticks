
#include <cassert>
#include "BBufSpec.hh"

#include "PLOG.hh"
#include "BRAP_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    BRAP_LOG_ ;

    BBufSpec bs = BBufSpec(1, NULL, 0, -1);

    assert(bs.id == 1 );
    assert(bs.ptr == NULL );
    assert(bs.num_bytes == 0 );
    assert(bs.target == -1 );



    return 0 ;
}

