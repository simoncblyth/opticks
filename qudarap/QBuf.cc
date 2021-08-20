
#include <cuda_runtime.h>
#include "scuda.h"
#include "squad.h"
#include "QBuf.hh"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
// quell warning: type attributes ignored after type is already defined [-Wattributes]
template struct QUDARAP_API QBuf<int>;
template struct QUDARAP_API QBuf<float>;
template struct QUDARAP_API QBuf<quad6>;
#pragma GCC diagnostic pop


