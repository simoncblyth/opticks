
#ifdef __clang__

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wunused-parameter"

#elif defined(__GNUC__) || defined(__GNUG__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"

#elif defined(_MSC_VER)

#pragma warning(push)
// nonstandard extension used: nameless struct/union  (from glm )
//#pragma warning( disable : 4201 )
// members needs to have dll-interface to be used by clients
//#pragma warning( disable : 4251 )
//
#endif


