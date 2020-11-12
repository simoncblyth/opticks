#include <iostream>
#include "Viz.hh"

Viz::Viz()
{
}

void Viz::command(const char* cmd)
{
    m_commands.push_back(cmd); 
    std::cout 
         << "Viz::command " 
         << m_commands.size() 
         << " [" << cmd << "]" 
         << std::endl 
         ; 
}



