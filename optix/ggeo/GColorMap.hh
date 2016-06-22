#pragma once

#include <map>
#include <string>

/*
*GColorMap* manages the association of named items with colors
*/

// hmm this is just a string string map, nothing special for color .. 
// replacing with npy-/Map<std::string, std::string> in new GPropertyLib approach 


#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GColorMap  {
   public:
       static GColorMap* load(const char* dir, const char* name);
   public:
       GColorMap();
       void dump(const char* msg="GColorMap::dump");

   public:
       void addItemColor(const char* iname, const char* color); 
       const char* getItemColor(const char* iname, const char* missing=NULL);

   private:
       void loadMaps(const char* idpath, const char* name);

   private:
       std::map<std::string, std::string>  m_iname2color ; 

};

#include "GGEO_TAIL.hh"

