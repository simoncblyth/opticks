#include <string>
#include "BList.hh"

const char* ini = "/tmp/BListTest.ini" ;
const char* json = "/tmp/BListTest.json" ;
typedef std::pair<std::string, unsigned int> SU ; 

void test_saveList()
{
   std::vector<SU> vp ; 
   vp.push_back(SU("hello",1));  
   vp.push_back(SU("hello",2));  // this replaces the first hello
   vp.push_back(SU("world",3));

   BList<std::string,unsigned int>::save(&vp, ini);
   BList<std::string,unsigned int>::save(&vp, json);
}

void test_loadList()
{
   std::vector<SU> vp ; 
   BList<std::string,unsigned int>::load(&vp, ini);
   BList<std::string,unsigned int>::dump(&vp, "loadList.ini");

   vp.clear(); 
   BList<std::string,unsigned int>::load(&vp, json);
   BList<std::string,unsigned int>::dump(&vp, "loadList.json");
}



int main()
{
    test_saveList();
    test_loadList();
    return 0 ; 
}

