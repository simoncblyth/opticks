#pragma once

#include "GItemIndex.hh"
#include "GGEO_API_EXPORT.hh"

class GGEO_API GSurfaceIndex : public GItemIndex  {
   public:
        static GSurfaceIndex* load(const char* idpath);
   public:
        GSurfaceIndex();

};


