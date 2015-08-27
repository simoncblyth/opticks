#include "GBuffer.hh"
#include "stdio.h"


void GBuffer::Summary(const char* msg)
{
    printf("%s NumBytes %u Ptr %p ItemSize %u NumElements_PerItem %u NumItems(NumBytes/ItemSize) %u NumElementsTotal (NumItems*NumElements) %u BufferId %d \n", 
          msg, 
          getNumBytes(), 
          getPointer(), 
          getItemSize(), 
          getNumElements(), 
          getNumItems(),
          getNumElementsTotal(),
          getBufferId() );
}


