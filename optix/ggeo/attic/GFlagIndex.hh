#pragma once

#include "GItemIndex.hh"

class GFlagIndex : public GItemIndex  {
   public:
        static GFlagIndex* load(const char* idpath);
   public:
        GFlagIndex();

};

inline GFlagIndex::GFlagIndex() : GItemIndex("GFlagIndex")
{
}

