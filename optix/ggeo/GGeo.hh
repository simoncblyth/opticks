#ifndef GGEO_H
#define GGEO_H

#include <vector>
class GSolid ; 

class GGeo {
   public:
      GGeo();
      virtual ~GGeo();

   public:
      void addSolid(GSolid* solid);

   private:
      std::vector<GSolid*> m_solids ; 

};

#endif


