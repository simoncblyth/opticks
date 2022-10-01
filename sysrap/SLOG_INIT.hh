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
#include <plog/Formatters/MessageOnlyFormatter.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Formatters/CsvFormatter.h>

//typedef plog::FuncMessageFormatter FMT ;     // useful to avoid dates and pids when comparing logs
//typedef plog::MessageOnlyFormatter FMT ;   // really minimal 
typedef plog::TxtFormatter         FMT ;   // default full format 
//typedef plog::CsvFormatter         FMT ;   // semicolon delimited full format  

#include "PlainFormatter.hh"

/**

SLOG\_INIT : logging macros
==============================


SLOG\_INIT macros are used in two situations:

* an executable main as a result of SLOG\_ or PLOT\_COLOR applied
  to the arguments

* package logger 


**/



#define SLOG_INIT0(level, app1, app2 ) \
{ \
    plog::IAppender* appender1 = app1 ? static_cast<plog::IAppender*>(app1) : NULL ; \
    plog::IAppender* appender2 = app2 ? static_cast<plog::IAppender*>(app2) : NULL ; \
    plog::Severity severity = static_cast<plog::Severity>(level) ; \
    plog::init( severity ,  appender1 ); \
    if(appender2) \
        plog::get()->addAppender(appender2) ; \
} \


#define SLOG_INIT(level, app1, app2 ) \
{ \
    plog::IAppender* appender1 = static_cast<plog::IAppender*>(app1) ; \
    plog::IAppender* appender2 = static_cast<plog::IAppender*>(app2) ; \
    plog::Severity severity = static_cast<plog::Severity>(level) ; \
    plog::init( severity ,  appender1 ); \
    if(appender2) \
        plog::get()->addAppender(appender2) ; \
} \



#define SLOG_INIT_(level, app1, app2, IDX ) \
{ \
    plog::IAppender* appender1 = static_cast<plog::IAppender*>(app1) ; \
    plog::IAppender* appender2 = static_cast<plog::IAppender*>(app2) ; \
    plog::Severity severity = static_cast<plog::Severity>(level) ; \
    plog::init<IDX>( severity ,  appender1 ); \
    if(appender2) \
        plog::get<IDX>()->addAppender(appender2) ; \
} \







#define SLOG_ECOLOR(name) \
{ \
    SLOG* _plog = new SLOG(name); \
    static plog::RollingFileAppender<FMT> fileAppender( _plog->filename, _plog->maxFileSize, _plog->maxFiles ); \
    static plog::ColorConsoleAppender<FMT> consoleAppender; \
    SLOG_INIT( _plog->level, &consoleAppender, &fileAppender ); \
} \

#define SLOG_COLOR(argc, argv) \
{ \
    SLOG* _plog = new SLOG(argc, argv); \
    static plog::RollingFileAppender<FMT> fileAppender( _plog->filename, _plog->maxFileSize, _plog->maxFiles ); \
    static plog::ColorConsoleAppender<FMT> consoleAppender; \
    SLOG_INIT( _plog->level, &consoleAppender, &fileAppender ); \
} \

#define SLOG_(argc, argv) \
{ \
    SLOG* _plog = new SLOG(argc, argv); \
    static plog::RollingFileAppender<FMT> fileAppender( _plog->filename, _plog->maxFileSize, _plog->maxFiles ); \
    static plog::ConsoleAppender<FMT> consoleAppender; \
    SLOG_INIT( _plog->level,  &consoleAppender, &fileAppender ); \
} \



#define SLOG_CHECK(msg) \
{ \
    LOG(fatal) << msg  ; \
    LOG(error) << msg  ; \
    LOG(warning) << msg  ; \
    LOG(info) << msg  ; \
    LOG(debug) << msg  ; \
    LOG(verbose) << msg  ; \
    LOG(verbose) << msg  ; \
} \





