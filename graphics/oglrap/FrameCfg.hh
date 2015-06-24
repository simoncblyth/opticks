#pragma once
#include "Cfg.hh"

template <class Listener>
class FrameCfg : public Cfg {
public:
   FrameCfg(const char* name, Listener* listener, bool live) 
       : 
       Cfg(name, live),
       m_bouncemax(9)   // one less than canonical maxrec of 10
   {   

       m_desc.add_options()
           ("version,v", "print version string") ;

       m_desc.add_options()
           ("help,h",    "print help message") ;

       m_desc.add_options()
           ("idpath,i",  "print idpath based on input envvars") ;


       // TODO: move the below to somewhere more appropriate

       m_desc.add_options()
           ("nogeocache,G",  "inhibit use of the geocache") ;

       m_desc.add_options()
           ("nooptix,O",  "inhibit use of OptiX") ;

       m_desc.add_options()
           ("scintillation,s",  "load scintillation gensteps") ;

       m_desc.add_options()
           ("cerenkov,c",  "load cerenkov gensteps") ;

       m_desc.add_options()
           ("alt,a",  "use alternative record renderer") ;

       m_desc.add_options()
           ("fullscreen,f",  "start in fullscreen mode") ;

       m_desc.add_options()
           ("tag",   boost::program_options::value<std::string>(&m_event_tag), "eventtag to load" );

       char bouncemax[128];
       snprintf(bouncemax,128, "Maximum number of boundary bounces, 0:to just generate. Default %d ", m_bouncemax);
       m_desc.add_options()
           ("bouncemax,b",  boost::program_options::value<int>(&m_bouncemax), bouncemax );

       ///////////////

       m_desc.add_options()
           ("config",   boost::program_options::value<std::string>(&m_configpath),
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
   std::string& getEventTag()
   {
       return m_event_tag ;
   }
   std::string& getLiveLine()
   {
       return m_liveline ;
   }
   int getBounceMax()
   {
        return m_bouncemax ; 
   }


private:
    std::string m_configpath ;
    std::string m_event_tag ;
    std::string m_liveline ;
    int         m_bouncemax ; 

};

