#include <vector>
#include <string>
#include "Ctrl.hh"

#define API  __attribute__ ((visibility ("default")))

class API MockViz : public Ctrl 
{
   public:
       MockViz(); 
       void command(const char* cmd); 
   private:
       std::vector<std::string> m_commands ; 
};


