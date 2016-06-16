#include "BJson.hh"

const char* path = "/tmp/jsonutilTest.json" ; 
const char* pathi = "/tmp/jsonutilTest.ini" ; 


void test_saveMap2()
{
   std::map<std::string, unsigned int> index ;
   index["/prefix/red"] = 1 ; 
   index["/prefix/green"] = 2 ; 
   index["/prefix/blue"] = 3 ; 

   BJson::saveMap<std::string, unsigned int>(index, path );
   BJson::saveMap<std::string, unsigned int>(index, pathi );
}

void test_loadMap2()
{
   std::map<std::string, unsigned int> index ;
   BJson::loadMap<std::string, unsigned int>(index, path );
   BJson::dumpMap<std::string, unsigned int>(index);
}



void test_saveMap()
{
   std::map<unsigned int, std::string> index ;
   index[0] = "hello0" ; 
   index[1] = "hello1" ; 
   index[10] = "hello10" ; 

   BJson::saveMap<unsigned int, std::string>(index, path );
}

void test_loadMap()
{
   std::map<unsigned int, std::string> index ;
   BJson::loadMap<unsigned int, std::string>(index, path );
   BJson::dumpMap<unsigned int, std::string>(index);
}

void test_loadIni()
{
   std::map<std::string, std::string> md ;
   BJson::loadMap<std::string, std::string>(md, "/tmp/g4_00.ini" );
   BJson::dumpMap<std::string, std::string>(md);
}


void test_saveList()
{

   typedef std::pair<std::string, unsigned int> SU ; 
   std::vector<SU> vp ; 
   vp.push_back(SU("hello",1));  
   vp.push_back(SU("hello",2));  // this replaces the first hello
   vp.push_back(SU("world",3));

   BJson::saveList(vp, "/tmp/list.ini");
   BJson::saveList(vp, "/tmp/list.json");
}

void test_loadList()
{
   typedef std::pair<std::string, unsigned int> SU ; 
   std::vector<SU> vp ; 
   BJson::loadList(vp, "/tmp/list.ini");
   BJson::dumpList(vp, "dumpList /tmp/list.ini");
}



int main()
{
    //test_saveMap();
    //test_loadMap();

    //test_saveList();
    //test_loadList();

    test_loadIni();

    return 0 ; 
}

