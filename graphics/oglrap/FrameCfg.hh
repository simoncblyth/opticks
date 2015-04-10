#pragma once
#include "Cfg.hh"

template <class Listener>
class FrameCfg : public Cfg {
public:
   FrameCfg(const char* name, Listener* listener, bool live) : Cfg(name, live) 
   {   

       m_desc.add_options()
           ("version,v", "print version string") ;

       m_desc.add_options()
           ("help,h",    "print help message") ;

       m_desc.add_options()
           ("config,c",   boost::program_options::value<std::string>(&m_configpath),
             "name of a file of a configuration.");

       m_desc.add_options()
           ("liveline",  boost::program_options::value<std::string>(&m_liveline),
               "string with spaces to be live parsed, as test of composed overrides");

       addOptionS<Listener>(listener, "size", "Comma delimited screen window coordinate width,height,window2pixel eg 1024,768,2  ");
       addOptionI<Listener>(listener, "dumpevent", "Control GLFW event dumping ");
   }   


   std::string& getConfigPath()
   {
       return m_configpath ;
   }
   std::string& getLiveLine()
   {
       return m_liveline ;
   }
   bool isHelp()
   {
       return m_vm.count("help") ;
   } 
   bool isAbort()
   {
       return m_vm.count("help") || m_vm.count("version") ; 
   } 

private:
    std::string m_configpath ;
    std::string m_liveline ;

};

