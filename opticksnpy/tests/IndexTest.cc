#include "Index.hpp"
#include <cassert>
#include "PLOG.hh"



// see ggeo-/GItemIndexTest  op --gitemindex " 

int main(int argc , char** argv )
{
   PLOG_(argc, argv);


   Index idx("IndexTest");
   idx.add("red",1);
   idx.add("green",2);
   idx.add("blue",3);

   assert(idx.getIndexSource("green") == 2 );

   int* ptr = idx.getSelectedPtr();
  
   for(unsigned i=0 ; i < idx.getNumKeys() ; i++ )
   { 
      *ptr = i ; 
      LOG(info) << std::setw(4) << i << " " << idx.getSelectedKey() ; 
   }


   return 0 ; 
}

