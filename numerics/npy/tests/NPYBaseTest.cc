#include "BLog.hh"

int main(int argc, char** argv)
{
   BLog nl("NPYBaseTest.log", "debug");
   nl.configure(argc, argv, "/tmp");

   LOG(info) << argv[0] ;

}
