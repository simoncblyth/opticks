#pragma once
#include <cstddef>

class GGeo ; 
class MTool ; 

class MFixer {
   public:
       MFixer(GGeo* ggeo);
       void fixMesh();
   private:
       void init();
   private:
       GGeo*   m_ggeo ;
       MTool*  m_tool ; 

};

inline MFixer::MFixer(GGeo* ggeo) : 
    m_ggeo(ggeo), 
    m_tool(NULL)
{
    init();
}

