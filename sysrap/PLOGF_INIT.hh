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

/**
PLOGF_INIT
=============

Color PLOG logging macros.

**/


#include <plog/Log.h>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Appenders/ConsoleAppender.h>

#define PLOGF_(argc, argv) \
{ \
    PLOG _plog(argc, argv) \
    static plog::RollingFileAppender<plog::TxtFormatter> fileAppender( _plog.filename, _plog.maxFileSize, _plog.maxFiles ); \
    static plog::ConsoleAppender<plog::TxtFormatter> consoleAppender; \
    plog::Severity severity = static_cast<plog::Severity>(_plog.level) ; \
    plog::init( severity , &fileAppender ).addAppender(&consoleAppender) ; \
}

#define PLOGF_CHECK(msg) \
{ \
    LOG(fatal) << msg  ; \
    LOG(error) << msg  ; \
    LOG(warning) << msg  ; \
    LOG(info) << msg  ; \
    LOG(debug) << msg  ; \
    LOG(verbose) << msg  ; \
} \





