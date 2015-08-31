#pragma once
#include "Cfg.hh"

template <class Listener>
class FrameCfg : public Cfg {
  public:
     FrameCfg(const char* name, Listener* listener, bool live);
  public:
     std::string& getConfigPath();
     std::string& getEventTag();
     std::string& getLiveLine();
     int          getBounceMax(); 
     int          getRecordMax(); 
     int          getTimeMax(); 
     int          getRepeatIndex(); 
     int          getModulo(); 
     int          getOverride(); 
     int          getDebugIdx(); 
private:
     void init();
private:
     Listener*   m_listener ; 
     std::string m_configpath ;
     std::string m_event_tag ;
     std::string m_liveline ;
     int         m_bouncemax ; 
     int         m_recordmax ; 
     int         m_timemax ; 
     int         m_repeatidx ; 
     int         m_modulo ; 
     int         m_override ; 
     int         m_debugidx ; 
};

template <class Listener>
inline FrameCfg<Listener>::FrameCfg(const char* name, Listener* listener, bool live) 
       : 
       Cfg(name, live),
       m_listener(listener),
       m_bouncemax(9),     
       m_recordmax(10),
       m_timemax(200),
       m_repeatidx(-1),
       m_modulo(-1),
       m_override(-1),
       m_debugidx(0)
{   
   init();  
}


template <class Listener>
inline void FrameCfg<Listener>::init()
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
       ("nopropagate,P",  "inhibit generation/propagation") ;

   m_desc.add_options()
       ("noevent,E",  "inhibit event handling") ;

   m_desc.add_options()
       ("nooptix,O",  "inhibit use of OptiX") ;

   m_desc.add_options()
       ("noviz,V",  "just generate, propagate and save : no visualization") ;

   m_desc.add_options()
       ("noindex,I",  "no photon/record indexing") ;

   m_desc.add_options()
       ("trivial",  "swap OptiX generate program with trivial standin for debugging") ;

   m_desc.add_options()
       ("geocenter",  "center view on geometry rather than the default genstep centering") ;

   m_desc.add_options()
       ("compute",  "COMPUTE mode, ie not INTEROP") ; 

   m_desc.add_options()
       ("scintillation,s",  "load scintillation gensteps") ;

   m_desc.add_options()
       ("cerenkov,c",  "load cerenkov gensteps") ;

   char modulo[128];
   snprintf(modulo,128, "Modulo scaledown input gensteps for speed/debugging, eg 10 to decimate. Values less than 1 disable scaledown. Default %d", m_modulo);
   m_desc.add_options()
       ("modulo",  boost::program_options::value<int>(&m_modulo), modulo );

   char override[128];
   snprintf(override,128, "Override photons to generate/propagate for debugging, eg 1 for a single photon. Values less than 1 disable any override. Default %d", m_override);
   m_desc.add_options()
       ("override",  boost::program_options::value<int>(&m_override), override );

   char debugidx[128];
   snprintf(debugidx,128, "Index of item eg Photon for debugging. Default %d", m_debugidx);
   m_desc.add_options()
       ("debugidx",  boost::program_options::value<int>(&m_debugidx), debugidx );


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



   // keeping bouncemax one less than recordmax is advantageous 
   // as bookeeping is then consistent between the photons and the records 
   // as this avoiding truncation of the records

   char recordmax[128];
   snprintf(recordmax,128, "Maximum number of photon records, 1:to minimize. Default %d ", m_recordmax);
   m_desc.add_options()
       ("recordmax,r",  boost::program_options::value<int>(&m_recordmax), recordmax );

   char timemax[128];
   snprintf(timemax,128, "Maximum time in nanoseconds. Default %d ", m_timemax);
   m_desc.add_options()
       ("timemax",  boost::program_options::value<int>(&m_timemax), timemax );

   char repeatidx[128];
   snprintf(repeatidx,128, "Repeat index used in development of instanced geometry, -1:flat full geometry. Default %d ", m_repeatidx);
   m_desc.add_options()
       ("repeatidx",  boost::program_options::value<int>(&m_repeatidx), repeatidx );

   ///////////////

   m_desc.add_options()
       ("config",   boost::program_options::value<std::string>(&m_configpath),
         "name of a file of a configuration.");

   m_desc.add_options()
       ("liveline",  boost::program_options::value<std::string>(&m_liveline),
           "string with spaces to be live parsed, as test of composed overrides");

   addOptionS<Listener>(m_listener, 
            "size", 
            "Comma delimited screen window coordinate width,height,window2pixel eg 1024,768,2  ");
   // this size is being overriden: 
   // the screen size is set by Frame::init using size from composition 

   addOptionI<Listener>(m_listener, 
            "dumpevent", 
            "Control GLFW event dumping ");
}   



template <class Listener>
inline std::string& FrameCfg<Listener>::getConfigPath()
{
    return m_configpath ;
}
template <class Listener>
inline std::string& FrameCfg<Listener>::getEventTag()
{
    return m_event_tag ;
}
template <class Listener>
inline std::string& FrameCfg<Listener>::getLiveLine()
{
    return m_liveline ;
}
template <class Listener>
inline int FrameCfg<Listener>::getBounceMax()
{
    return m_bouncemax ; 
}
template <class Listener>
inline int FrameCfg<Listener>::getRecordMax()
{
    return m_recordmax ; 
}
template <class Listener>
inline int FrameCfg<Listener>::getTimeMax()
{
    return m_timemax ; 
}
template <class Listener>
inline int FrameCfg<Listener>::getRepeatIndex()
{
    return m_repeatidx ; 
}
template <class Listener>
inline int FrameCfg<Listener>::getModulo()
{
    return m_modulo ; 
}
template <class Listener>
inline int FrameCfg<Listener>::getOverride()
{
    return m_override ; 
}
template <class Listener>
inline int FrameCfg<Listener>::getDebugIdx()
{
    return m_debugidx ; 
}






