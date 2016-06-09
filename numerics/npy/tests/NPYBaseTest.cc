#include "BLog.hh"

int main(int argc, char** argv)
{
   BLog nl(argc, argv);

   LOG(info) << argv[0] ;

}
