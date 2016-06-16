#pragma once

#include <boost/program_options.hpp>
#include <string>
#include <vector>
#include <cstdio>
#include <boost/bind.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/algorithm/string.hpp>

/*
Listener classes need to provide a methods::

   void configureF(const char* name, std::vector<float> values);
   void configureI(const char* name, std::vector<int> values);
   void configureS(const char* name, std::vector<std::string> values);
 
which is called whenever the option parsing methods are called. 
Typically the last value in the vector should be used to call the Listeners 
setter method as selected by the name.

*/
#include "BRAP_API_EXPORT.hh"

class BRAP_API BCfg {

public:
     void dumpTree(const char* msg="BCfg::dumpTree");
private:
     void dumpTree_(unsigned int depth=0);

protected:
    boost::program_options::variables_map       m_vm;
    boost::program_options::options_description m_desc ; 

    // binding to overloaded methods is problematic : so spell it out 

    template <class Listener>
    void addOptionF(Listener* listener, const char* name, const char* description);

    template <class Listener>
    void addOptionI(Listener* listener, const char* name, const char* description);

    template <class Listener>
    void addOptionS(Listener* listener, const char* name, const char* description);

public:
    BCfg(const char* name, bool live);
    bool isLive();
    const char* getName();
    bool hasOpt(const char* opt);   

public:
    // holding others 
    void add(BCfg* cfg);
    bool containsOthers();
    unsigned int getNumOthers();
    BCfg* getOther(unsigned int index);
    BCfg* findOther(const char* name);
    BCfg* operator [](const char* name);

public:
    void setVerbose(bool verbose=true);
    boost::program_options::options_description& getDesc();
    std::string getDescString();

    const std::string& getCommandLine(); 
    virtual void commandline(int argc, char** argv);
    void liveline(const char* line);
    void configfile(const char* path);

    std::vector<std::string> parse_commandline(int argc, char** argv, bool verbose=false);
    std::vector<std::string> parse_configfile(const char* path);
    std::vector<std::string> parse_liveline(const char* line);
    std::vector<std::string> parse_tokens(std::vector<std::string>& tokens);

    virtual void dump(const char* msg);

    static void dump(std::vector<std::string>& ss, const char* msg );
    static void dump(boost::program_options::options_description& desc, const char* msg);
    static void dump(boost::program_options::variables_map&       vm, const char* msg);
    static void dump(boost::program_options::parsed_options& opts, const char* msg );

    bool hasError(); 
    std::string getErrorMessage(); 
private:
    const char*       m_name ;    
    std::vector<BCfg*> m_others ; 
    std::string       m_commandline ; 
    bool              m_live ; 
    bool              m_error ; 
    std::string       m_error_message ; 
    bool              m_verbose ; 

};


inline void BCfg::setVerbose(bool verbose)
{
    m_verbose = verbose ; 
}


inline bool BCfg::hasError()
{
    return m_error ; 
}

inline std::string BCfg::getErrorMessage()
{
    return m_error_message ; 
}


inline const std::string& BCfg::getCommandLine()
{
    return m_commandline ; 
}

inline bool BCfg::hasOpt(const char* opt)
{ 
   std::vector<std::string> elem;
   boost::split(elem,opt,boost::is_any_of("|")); 
   unsigned int count(0);
   for(unsigned int i=0 ; i < elem.size() ; i++)
   { 
      count += m_vm.count(elem[i]);
   }
   return count > 0 ; 
}   


inline const char* BCfg::getName()
{
    return m_name ; 
}

inline bool BCfg::isLive()
{
    return m_live ; 
}





template <class Listener>
void BCfg::addOptionF(Listener* listener, const char* name, const char* description )
{
        m_desc.add_options()(name, 
                             boost::program_options::value<std::vector<float> >()
                                ->composing()
                                ->notifier(boost::bind(&Listener::configureF, listener, name, _1)), 
                             description) ;
}

template <class Listener>
void BCfg::addOptionI(Listener* listener, const char* name, const char* description )
{
        m_desc.add_options()(name, 
                             boost::program_options::value<std::vector<int> >()
                                ->composing()
                                ->notifier(boost::bind(&Listener::configureI, listener, name, _1)), 
                             description) ;
}


template <class Listener>
void BCfg::addOptionS(Listener* listener, const char* name, const char* description )
{
        if(m_verbose)
        {
             printf("BCfg::addOptionS %s %s \n", name, description);
        }
        m_desc.add_options()(name, 
                             boost::program_options::value<std::vector<std::string> >()
                                ->composing()
                                ->notifier(boost::bind(&Listener::configureS, listener, name, _1)), 
                             description) ;
}



