#include <iostream>
#include "SMockViz.hh"
#include "SLOG.hh"

const plog::Severity SMockViz::LEVEL = SLOG::EnvLevel("SMockViz", "DEBUG"); 

SMockViz::SMockViz()
{
}

void SMockViz::command(const char* cmd)
{
    m_commands.push_back(cmd); 
    LOG(info)
         << m_commands.size() 
         << " [" << cmd << "]" 
         ; 

}



