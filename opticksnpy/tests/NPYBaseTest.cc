#include "PLOG.hh"

int main(int argc, char** argv)
{
   PLOG_(argc, argv);

   LOG(info) << argv[0] ;

}
