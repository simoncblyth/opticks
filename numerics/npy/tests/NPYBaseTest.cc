#include "BLog.hh"

int main(int argc, char** argv)
{
   NLog nl("NPYBaseTest.log", "debug");
   nl.configure(argc, argv, "/tmp");

   LOG(info) << argv[0] ;

}
