#include <iostream>
#include "MockViz.hh"

MockViz::MockViz()
{
}

void MockViz::command(const char* cmd)
{
    m_commands.push_back(cmd); 
    std::cout 
         << "MockViz::command " 
         << m_commands.size() 
         << " [" << cmd << "]" 
         << std::endl 
         ; 
}



