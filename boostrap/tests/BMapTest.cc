#include <string>

#include "BMap.hh"

#include "BRAP_LOG.hh"
#include "PLOG.hh"

const char* pathSU = "$TMP/BMapTestSU.json" ; 
const char* pathUS = "$TMP/BMapTestUS.json" ;
const char* pathSUi = "$TMP/BMapTestSU.ini" ; 
const char* pathSSi = "$TMP/BMapTestSS.ini" ; 


void test_saveMapSU()
{
   std::map<std::string, unsigned int> index ;
   index["/prefix/red"] = 1 ; 
   index["/prefix/green"] = 2 ; 
   index["/prefix/blue"] = 3 ; 

   BMap<std::string,unsigned int>::save(&index, pathSU );
   BMap<std::string,unsigned int>::save(&index, pathSUi );
}
void test_loadMapSU()
{
   std::map<std::string, unsigned int> index ;
   BMap<std::string, unsigned int>::load(&index, pathSU );
   BMap<std::string, unsigned int>::dump(&index,"loadMapSU");
}


void test_saveMapUS()
{
   std::map<unsigned int, std::string> index ;
   index[0] = "hello0" ; 
   index[1] = "hello1" ; 
   index[10] = "hello10" ; 
   BMap<unsigned int, std::string>::save(&index, pathUS );
}
void test_loadMapUS()
{
   std::map<unsigned int, std::string> index ;
   BMap<unsigned int, std::string>::load(&index, pathUS );
   BMap<unsigned int, std::string>::dump(&index,"loadMapUS");
}


void test_saveIni()
{
   std::map<std::string, std::string> md ;
   md["red"] = "a" ; 
   md["green"] = "b" ; 
   md["blue"] = "c" ; 

   BMap<std::string, std::string>::dump(&md, "saveIni");
   BMap<std::string, std::string>::save(&md, pathSSi );
}

void test_loadIni()
{
   std::map<std::string, std::string> md ;
   BMap<std::string, std::string>::load(&md, pathSSi );
   BMap<std::string, std::string>::dump(&md, "loadIni");
}





int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    BRAP_LOG__ ;

    LOG(info) << argv[0] ;
  
    test_saveMapSU();
    test_loadMapSU();

    test_saveMapUS();
    test_loadMapUS();

    test_saveIni();
    test_loadIni();

    return 0 ; 
}

