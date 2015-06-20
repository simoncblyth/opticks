#pragma once

#include "GItemIndex.hh"

class GSurfaceIndex : public GItemIndex  {
   public:
        static GSurfaceIndex* load(const char* idpath);
   public:
        GSurfaceIndex();

};

inline GSurfaceIndex::GSurfaceIndex() : GItemIndex("GSurfaceIndex")
{
}

