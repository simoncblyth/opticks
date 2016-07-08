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

// warning C4244: 'argument': conversion from 'double' to 'float', possible loss of data // from CLHEP/Random headers
#pragma warning( disable : 4244 )

#endif

