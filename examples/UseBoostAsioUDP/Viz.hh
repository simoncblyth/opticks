#include <vector>
#include <string>
#include "SCtrl.hh"

class Viz : public SCtrl 
{
   public:
       Viz(); 
       void command(const char* cmd); 
   private:
       std::vector<std::string> m_commands ; 
};


