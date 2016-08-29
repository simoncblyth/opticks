#include <plog/Log.h>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Appenders/ConsoleAppender.h>

/*

PLOG_INIT macros are used in two situations:

* an executable main as a result of PLOG_ or PLOT_COLOR applied
  to the arguments

* package logger 


*/



#define PLOG_INIT(level, app1, app2 ) \
{ \
    plog::IAppender* appender1 = app1 ? static_cast<plog::IAppender*>(app1) : NULL ; \
    plog::IAppender* appender2 = app2 ? static_cast<plog::IAppender*>(app2) : NULL ; \
    plog::Severity severity = static_cast<plog::Severity>(level) ; \
    plog::init( severity ,  appender1 ); \
    if(appender2) \
        plog::get()->addAppender(appender2) ; \
} \


#define PLOG_COLOR(argc, argv) \
{ \
    PLOG _plog(argc, argv); \
    static plog::RollingFileAppender<plog::TxtFormatter> fileAppender( _plog.logpath, _plog.logmax); \
    static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender; \
    PLOG_INIT( _plog.level, &consoleAppender, &fileAppender ); \
} \

#define PLOG_(argc, argv) \
{ \
    PLOG _plog(argc, argv); \
    static plog::RollingFileAppender<plog::TxtFormatter> fileAppender( _plog.logpath, _plog.logmax); \
    static plog::ConsoleAppender<plog::TxtFormatter> consoleAppender; \
    PLOG_INIT( _plog.level,  &consoleAppender, &fileAppender ); \
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





