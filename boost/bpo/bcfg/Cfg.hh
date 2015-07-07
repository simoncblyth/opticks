#pragma once

#include <boost/program_options.hpp>
#include <string>
#include <vector>
#include <boost/bind.hpp>
#include <boost/assign/list_of.hpp>


/*
Listener classes need to provide a methods::

   void configureF(const char* name, std::vector<float> values);
   void configureI(const char* name, std::vector<int> values);
   void configureS(const char* name, std::vector<std::string> values);
 
which is called whenever the option parsing methods are called. 
Typically the last value in the vector should be used to call the Listeners 
setter method as selected by the name.

*/


class Cfg {

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
    Cfg(const char* name, bool live);
    bool isLive();
    const char* getName();
    bool hasOpt(const char* opt);   
    bool isAbort();

public:
    // holding others 
    void add(Cfg* cfg);
    bool containsOthers();
    unsigned int getNumOthers();
    Cfg* getOther(unsigned int index);
    Cfg* findOther(const char* name);
    Cfg* operator [](const char* name);

public:
    boost::program_options::options_description& getDesc();
    std::string getDescString();

    const std::string& getCommandLine(); 
    void commandline(int argc, char** argv);
    void liveline(const char* line);
    void configfile(const char* path);

    std::vector<std::string> parse_commandline(int argc, char** argv);
    std::vector<std::string> parse_configfile(const char* path);
    std::vector<std::string> parse_liveline(const char* line);
    std::vector<std::string> parse_tokens(std::vector<std::string>& tokens);

    virtual void dump(const char* msg);

    static void dump(std::vector<std::string>& ss, const char* msg );
    static void dump(boost::program_options::options_description& desc, const char* msg);
    static void dump(boost::program_options::variables_map&       vm, const char* msg);
    static void dump(boost::program_options::parsed_options& opts, const char* msg );

private:
    const char* m_name ;    
    std::vector<Cfg*> m_others ; 
    std::string m_commandline ; 
    bool m_live ; 

};

inline const std::string& Cfg::getCommandLine()
{
    return m_commandline ; 
}

inline bool Cfg::hasOpt(const char* opt)
{ 
   return m_vm.count(opt) ; 
}   

inline bool Cfg::isAbort()
{
    // TODO: eliminate this by supporting comma delimited hasOpt checking 
    return hasOpt("help") || hasOpt("version") || hasOpt("idpath") ; 
}


template <class Listener>
void Cfg::addOptionF(Listener* listener, const char* name, const char* description )
{
        m_desc.add_options()(name, 
                             boost::program_options::value<std::vector<float> >()
                                ->composing()
                                ->notifier(boost::bind(&Listener::configureF, listener, name, _1)), 
                             description) ;
}

template <class Listener>
void Cfg::addOptionI(Listener* listener, const char* name, const char* description )
{
        m_desc.add_options()(name, 
                             boost::program_options::value<std::vector<int> >()
                                ->composing()
                                ->notifier(boost::bind(&Listener::configureI, listener, name, _1)), 
                             description) ;
}


template <class Listener>
void Cfg::addOptionS(Listener* listener, const char* name, const char* description )
{
        m_desc.add_options()(name, 
                             boost::program_options::value<std::vector<std::string> >()
                                ->composing()
                                ->notifier(boost::bind(&Listener::configureS, listener, name, _1)), 
                             description) ;
}



