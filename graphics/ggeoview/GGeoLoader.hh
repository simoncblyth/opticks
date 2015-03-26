#ifndef GGEOLOADER_H
#define GGEOLOADER_H

#include "stdlib.h"

#include "AssimpWrap/AssimpGeometry.hh"
#include "AssimpWrap/AssimpTree.hh"
#include "AssimpWrap/AssimpNode.hh"
#include "AssimpWrap/AssimpGGeo.hh"

#include "GGeo.hh"
#include "GMergedMesh.hh"


class GGeoLoader {
  public:
     GGeoLoader()
     {
          const char* geokey = getenv("GGEOVIEW_GEOKEY");
          const char* path = getenv(geokey);

          const char* query = getenv("GGEOVIEW_QUERY");
          AssimpGeometry ageo(path);
          ageo.import();
          AssimpSelection* selection = ageo.select(query);

          AssimpGGeo agg(ageo.getTree(), selection); 
          const char* ggctrl = getenv("GGEOVIEW_CTRL");
          m_ggeo = agg.convert(ggctrl);
     }
     virtual ~GGeoLoader()
     {
     }

     GGeo* getGGeo()
     {
         return m_ggeo ;  
     }

  private: 
     GGeo* m_ggeo ; 
};


#endif
