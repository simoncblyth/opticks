#include <vector>
#include <string>

#include "plog/Severity.h"
#include "SCtrl.hh"

#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API SMockViz : public SCtrl 
{
   private:
       static const plog::Severity LEVEL ; 
   public:
       SMockViz(); 
       void command(const char* cmd); 
   private:
       std::vector<std::string> m_commands ; 
};


