#pragma once

#if defined(WITH_SCOMPLEX)
    #include "scomplex.h"
    namespace opticks {
        template <typename T> using complex = opticks::scomplex<T>;
        // scomplex functions (norm, conj, etc.) are already defined inside namespace opticks
    }
#elif defined(WITH_THRUST)
    #include <thrust/complex.h>
    namespace opticks {
        template <typename T> using complex = thrust::complex<T>;
        using thrust::norm;
        using thrust::conj;
        using thrust::exp;
        using thrust::sqrt;
        using thrust::sin;
        using thrust::cos;
    }
#else
    #include <complex>
    #include <cmath>
    namespace opticks {
        template <typename T> using complex = std::complex<T>;
        using std::norm;
        using std::conj;
        using std::exp;
        using std::sqrt;
        using std::sin;
        using std::cos;
    }
#endif
