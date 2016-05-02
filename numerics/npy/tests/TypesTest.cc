#include "Types.hpp"

#include "stdlib.h"
#include "assert.h"
#include "stdio.h"

int main(int argc, char** argv)
{

    const char* idpath = getenv("IDPATH");

    Types types ; 
    types.dumpFlags();

   // material names have moved to GItemList control see $IDPATH/GItemList/GMaterialLib.txt
   // lower level version in NPropNames
    types.readMaterials(idpath, "GMaterialIndex");
    types.dumpMaterials();

    types.getHistoryStringTest();
    types.getMaterialStringTest();
  

    return 0 ;
}

