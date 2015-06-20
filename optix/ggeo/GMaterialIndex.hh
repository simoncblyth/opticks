#pragma once

#include "GItemIndex.hh"

class GMaterialIndex : public GItemIndex  {
   public:
        static GMaterialIndex* load(const char* idpath);
   public:
        GMaterialIndex();

};

inline GMaterialIndex::GMaterialIndex() : GItemIndex("GMaterialIndex")
{
}

