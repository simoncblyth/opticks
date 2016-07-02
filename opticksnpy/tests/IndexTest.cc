#include "Index.hpp"
#include <cassert>
#include <iostream>

// see ggeo-/GItemIndexTest  op --gitemindex " 

int main(int , char** )
{

   Index idx("IndexTest");
   idx.add("red",1);
   idx.add("green",2);
   idx.add("blue",3);

   assert(idx.getIndexSource("green") == 2 );


   return 0 ; 
}

