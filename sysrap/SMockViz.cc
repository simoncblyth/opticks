#include <iostream>
#include "SMockViz.hh"
#include "PLOG.hh"

const plog::Severity SMockViz::LEVEL = PLOG::EnvLevel("SMockViz", "DEBUG"); 

SMockViz::SMockViz()
{
}


/**
There seems to be a problem with PLOG logging when this gets 
called from within boost::asio handlers.  Only the std::cout 
deigns to appear.
**/

void SMockViz::command(const char* cmd)
{
    m_commands.push_back(cmd); 


    LOG(LEVEL)
         << m_commands.size() 
         << " [" << cmd << "]" 
         ; 

    LOG(info)
         << m_commands.size() 
         << " [" << cmd << "]" 
         ; 

    std::cout
         << m_commands.size() 
         << " [" << cmd << "]" 
         << std::endl 
         ; 

}



