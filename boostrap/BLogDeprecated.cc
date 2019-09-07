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

#define DBG 1
#include <string>
#include <iostream>

#ifdef _MSC_VER
// signed/unsigned mismatch
#pragma warning( disable : 4018 )
#endif


#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include "boost/log/utility/setup.hpp"
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

// brap-
#include "BFile.hh"
#include "BLog.hh"
#include "BSys.hh"

BLog::BLog(int argc, char** argv)
   :
     m_argc(argc),
     m_argv(argv),

     m_loglevel(0),
     m_logname(NULL),
     m_logdir(NULL),

     m_nogeocache(false),
     m_pause(false),
     m_exitpause(false),
     m_addfile(false)
{
     init();
}

int BLog::getLevel()
{
    return m_loglevel ; 
}

BLog::~BLog()
{
    if(m_exitpause) BSys::WaitForInput("Blog::~BLog exit-pausing...");
}


void BLog::parse(int argc, char** argv )
{
    std::cerr << "BLog::parse" 
              << " argc " << argc 
              << std::endl 
              ; 

    if(argc > 0)
    {
        std::string stem = BFile::Stem(argv[0]);
        std::string logname(stem) ;
        logname += ".log" ;
        setName(logname.c_str()); 
        // logname derived from basename of executable
    }   

    const char* ll = "INFO" ;   
    for(int i=1 ; i < argc ; ++i )
    {
        if(strcmp(argv[i], "--trace")==0)   ll = "TRACE" ; 
        if(strcmp(argv[i], "--debug")==0)   ll = "DEBUG" ; 
        if(strcmp(argv[i], "--info")==0)    ll = "INFO" ; 
        if(strcmp(argv[i], "--warning")==0) ll = "WARNING" ; 
        if(strcmp(argv[i], "--error")==0)   ll = "ERROR" ; 
        if(strcmp(argv[i], "--fatal")==0)   ll = "FATAL" ; 

        if(i < argc - 1 && strcmp(argv[i], "--logdir")==0) setDir(argv[i+1]) ;


        if(strcmp(argv[i], "-G")==0)              m_nogeocache = true ; 
        if(strcmp(argv[i], "--nogeocache")==0)    m_nogeocache = true ; 
        if(strcmp(argv[i], "--pause")==0)         m_pause = true ;    
        if(strcmp(argv[i], "--exitpause")==0)     m_exitpause = true ;    
    }

    m_loglevel = SeverityLevel(ll);

    std::cerr << "BLog::parse"
              << " loglevel " << m_loglevel 
              << std::endl ; 
}

int BLog::SeverityLevel(const char* ll)
{
    int bll = boost::log::trivial::info ;
    std::string level(ll);
    if(level.compare("TRACE") == 0) bll = boost::log::trivial::trace ; 
    if(level.compare("DEBUG") == 0) bll = boost::log::trivial::debug ; 
    if(level.compare("INFO") == 0)  bll = boost::log::trivial::info ; 
    if(level.compare("WARNING") == 0)  bll = boost::log::trivial::warning ; 
    if(level.compare("ERROR") == 0)  bll = boost::log::trivial::error ; 
    if(level.compare("FATAL") == 0)  bll = boost::log::trivial::fatal ; 
    return static_cast<int>(bll); 
}


void BLog::setName(const char* logname)
{
    if(logname) m_logname = strdup(logname) ;
}
void BLog::setDir(const char* logdir)
{
    if(logdir) m_logdir = strdup(logdir) ;
    addFileLog();
}


void BLog::init()
{
    parse(m_argc, m_argv);
    initialize(NULL) ;

    if(m_logdir) addFileLog();

    if(m_pause) BSys::WaitForInput("Blog::init pausing...");
}

void BLog::initialize(void* whatever)
{
    Initialize(whatever, m_loglevel);
}

void BLog::Initialize(void* whatever, int level)
{
    boost::log::core::get()->set_filter
    (    
        boost::log::trivial::severity >= level
    ); 
/*
    boost::log::add_common_attributes();
    boost::log::register_simple_formatter_factory< boost::log::trivial::severity_level, char >("Severity");  
    boost::log::add_console_log(
        std::cerr, 
        boost::log::keywords::format = "[%TimeStamp%]:%Severity%: %Message%",
        boost::log::keywords::auto_flush = true 
    );   
*/
}


void BLog::addFileLog()
{
    if(m_logname && m_logdir && !m_addfile )
    {
       m_addfile = true ; 

       BFile::CreateDir(m_logdir);
       std::string logpath = BFile::FormPath(m_logdir, m_logname) ;

       boost::log::add_file_log(
              logpath,
              boost::log::keywords::format = "[%TimeStamp%]:%Severity%: %Message%"
             );


#ifdef DBG
       std::cerr  << "BLog::addFileLog"
                  << " logpath " << logpath 
                  << std::endl 
                  ;

#endif
    }
}



