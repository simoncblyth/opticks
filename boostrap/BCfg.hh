/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

#include <string>
#include <vector>
#include <cstdio>

#include "plog/Severity.h"


#ifdef __clang__
#pragma GCC visibility push(default)
#endif

#include <boost/bind.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>

#ifdef __clang__
#pragma GCC visibility pop
#endif

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
#include "BRAP_HEAD.hh"


#ifdef _MSC_VER
// m_vm m_desc m_others m_commandline m_error_message needs dll-interface
#pragma warning( disable : 4251 )
#endif


class BRAP_API BCfg {
private:
     static const plog::Severity LEVEL ; 
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
    bool hasOpt(const char* opt) const ;   

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




template <class Listener>
inline void BCfg::addOptionF(Listener* listener, const char* name, const char* description )
{
        m_desc.add_options()(name, 
                             boost::program_options::value<std::vector<float> >()
                                ->composing()
                                ->notifier(boost::bind(&Listener::configureF, listener, name, _1)), 
                             description) ;
}

template <class Listener>
inline void BCfg::addOptionI(Listener* listener, const char* name, const char* description )
{
        m_desc.add_options()(name, 
                             boost::program_options::value<std::vector<int> >()
                                ->composing()
                                ->notifier(boost::bind(&Listener::configureI, listener, name, _1)), 
                             description) ;
}


template <class Listener>
inline void BCfg::addOptionS(Listener* listener, const char* name, const char* description )
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


#include "BRAP_TAIL.hh"

