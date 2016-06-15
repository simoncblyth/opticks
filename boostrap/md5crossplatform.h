#include <boost/predef.h>

#if defined(BOOST_OS_APPLE)

#   pragma message("BOOST_OS_APPLE")
#   define COMMON_DIGEST_FOR_OPENSSL
#   include <CommonCrypto/CommonDigest.h>
#   define SHA1 CC_SHA1

#elif defined(BOOST_OS_WINDOWS)

#   pragma message("BOOST_OS_WINDOWS")
#   include "md5.h"

#elif defined(BOOST_OS_LINUX)

#   pragma message("BOOST_OS_LINUX")
#   include <openssl/md5.h>

#endif

