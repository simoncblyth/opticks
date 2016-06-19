#include "Types.hpp"

#include <cstdlib>
#include <cassert>
#include <cstdio>

int main(int,char**)
{

    const char* idpath = getenv("IDPATH");

    Types types ; 
    types.dumpFlags();

   // material names have moved to GItemList control see $IDPATH/GItemList/GMaterialLib.txt
   // lower level version in NPropNames


    types.readMaterials(idpath, "GMaterialLib"); // sets the Index
    types.dumpMaterials();

    types.getHistoryStringTest();
    types.getMaterialStringTest();
  

    return 0 ;
}

