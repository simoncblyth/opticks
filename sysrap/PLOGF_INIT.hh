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





