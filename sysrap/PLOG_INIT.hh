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

#include <plog/Log.h>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Appenders/ConsoleAppender.h>
#include <plog/Formatters/FuncMessageFormatter.h>

#include "PlainFormatter.hh"

/**

PLOG\_INIT : logging macros
==============================


PLOG\_INIT macros are used in two situations:

* an executable main as a result of PLOG\_ or PLOT\_COLOR applied
  to the arguments

* package logger 


**/



#define PLOG_INIT0(level, app1, app2 ) \
{ \
    plog::IAppender* appender1 = app1 ? static_cast<plog::IAppender*>(app1) : NULL ; \
    plog::IAppender* appender2 = app2 ? static_cast<plog::IAppender*>(app2) : NULL ; \
    plog::Severity severity = static_cast<plog::Severity>(level) ; \
    plog::init( severity ,  appender1 ); \
    if(appender2) \
        plog::get()->addAppender(appender2) ; \
} \


#define PLOG_INIT(level, app1, app2 ) \
{ \
    plog::IAppender* appender1 = static_cast<plog::IAppender*>(app1) ; \
    plog::IAppender* appender2 = static_cast<plog::IAppender*>(app2) ; \
    plog::Severity severity = static_cast<plog::Severity>(level) ; \
    plog::init( severity ,  appender1 ); \
    if(appender2) \
        plog::get()->addAppender(appender2) ; \
} \





#define PLOG_COLOR(argc, argv) \
{ \
    PLOG* _plog = new PLOG(argc, argv); \
    static plog::RollingFileAppender<plog::TxtFormatter> fileAppender( _plog->filename, _plog->maxFileSize, _plog->maxFiles ); \
    static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender; \
    PLOG_INIT( _plog->level, &consoleAppender, &fileAppender ); \
} \

#define PLOG_(argc, argv) \
{ \
    PLOG* _plog = new PLOG(argc, argv); \
    static plog::RollingFileAppender<plog::TxtFormatter> fileAppender( _plog->filename, _plog->maxFileSize, _plog->maxFiles ); \
    static plog::ConsoleAppender<plog::TxtFormatter> consoleAppender; \
    PLOG_INIT( _plog->level,  &consoleAppender, &fileAppender ); \
} \



#define PLOG_CHECK(msg) \
{ \
    LOG(fatal) << msg  ; \
    LOG(error) << msg  ; \
    LOG(warning) << msg  ; \
    LOG(info) << msg  ; \
    LOG(debug) << msg  ; \
    LOG(verbose) << msg  ; \
    LOG(verbose) << msg  ; \
} \





