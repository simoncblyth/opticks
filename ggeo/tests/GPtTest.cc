// TEST=GPtTest om-t

#include "OPTICKS_LOG.hh"
#include "GPt.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    GPt* pt = new GPt(101, "red" ); 

    LOG(info) << pt->desc(); 


    return 0 ;
}

