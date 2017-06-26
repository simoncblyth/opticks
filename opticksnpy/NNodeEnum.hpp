#pragma once


typedef enum
{  
   FRAME_MODEL, 
   FRAME_LOCAL, 
   FRAME_GLOBAL 

} NNodeFrameType ;

#include "NPY_API_EXPORT.hh"

class NPY_API NNodeEnum
{
    public:
        static const char* FRAME_MODEL_ ;
        static const char* FRAME_LOCAL_;
        static const char* FRAME_GLOBAL_ ;
        static const char* FrameType(NNodeFrameType fr);

};
