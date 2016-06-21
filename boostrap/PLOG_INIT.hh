#include <plog/Log.h>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Appenders/ConsoleAppender.h>


#define PLOG_INIT(whatever, level ) \
{ \
    plog::IAppender* appender = static_cast<plog::IAppender*>(whatever) ; \
    plog::Severity severity = static_cast<plog::Severity>(level) ; \
    plog::init( severity ,  appender ); \
} \

#define PLOG_COLOR(argc, argv) \
{ \
    static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender; \
    PLOG_INIT(  &consoleAppender, PLOG(argc,argv).level ); \
} \

#define PLOG_(argc, argv) \
{ \
    static plog::ConsoleAppender<plog::TxtFormatter> consoleAppender; \
    PLOG_INIT(  &consoleAppender, PLOG(argc,argv).level ); \
} \

#define PLOG_CHECK(msg) \
{ \
    LOG(fatal) << msg  ; \
    LOG(error) << msg  ; \
    LOG(warning) << msg  ; \
    LOG(info) << msg  ; \
    LOG(debug) << msg  ; \
    LOG(trace) << msg  ; \
    LOG(verbose) << msg  ; \
} \





