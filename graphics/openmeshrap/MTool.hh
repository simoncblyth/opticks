#pragma once

class GMesh ; 

class MTool {
   public:
       MTool();
       static GMesh* joinSplitUnion(GMesh* mesh, const char* config);
   public:
       unsigned int countMeshComponents(GMesh* gm);

};


inline MTool::MTool()
{
}

