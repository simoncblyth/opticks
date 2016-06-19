#include "NTxt.hpp"

#include <cstdlib>
#include <cstring>


int main(int, char** )
{
    char* idp = getenv("IDPATH") ;
    char path[256];
    snprintf(path, 256, "%s/GItemList/GMaterialLib.txt", idp );

    NTxt txt(path);
    txt.read();
 


    return 0 ; 
}
