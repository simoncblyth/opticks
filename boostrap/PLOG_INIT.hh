#include <plog/Log.h>

#define PLOG_INIT(whatever, level ) \
{ \
    plog::IAppender* appender = static_cast<plog::IAppender*>(whatever) ; \
    plog::Severity severity = static_cast<plog::Severity>(level) ; \
    plog::init( severity ,  appender ); \
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





