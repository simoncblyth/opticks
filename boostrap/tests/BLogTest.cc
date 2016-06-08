#include "BLog.hh"


int main(int argc, char** argv)
{
    BLog bl("BLogTest.log", "info");

    bl.configure(argc, argv);


    return 0 ; 
}
