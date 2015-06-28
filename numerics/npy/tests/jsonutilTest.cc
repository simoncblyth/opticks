#include "jsonutil.hpp"

const char* path = "/tmp/jsonutilTest.json" ; 
const char* pathi = "/tmp/jsonutilTest.ini" ; 


void test_saveMap2()
{
   std::map<std::string, unsigned int> index ;
   index["/prefix/red"] = 1 ; 
   index["/prefix/green"] = 2 ; 
   index["/prefix/blue"] = 3 ; 

   saveMap<std::string, unsigned int>(index, path );
   saveMap<std::string, unsigned int>(index, pathi );
}

void test_loadMap2()
{
   std::map<std::string, unsigned int> index ;
   loadMap<std::string, unsigned int>(index, path );
   dumpMap<std::string, unsigned int>(index);
}



void test_saveMap()
{
   std::map<unsigned int, std::string> index ;
   index[0] = "hello0" ; 
   index[1] = "hello1" ; 
   index[10] = "hello10" ; 

   saveMap<unsigned int, std::string>(index, path );
}

void test_loadMap()
{
   std::map<unsigned int, std::string> index ;
   loadMap<unsigned int, std::string>(index, path );
   dumpMap<unsigned int, std::string>(index);
}

int main()
{
    //test_saveMap();
    //test_loadMap();

    test_saveMap2();
    test_loadMap2();



    return 0 ; 
}

