#include "BLog.hh"


int main(int argc, char** argv)
{
    BLog bl(argc, argv);

    LOG(info) << argv[0] << " before " ;

    bl.setDir("/tmp");

    LOG(info) << argv[0] << " after " ;

    return 0 ; 
}
