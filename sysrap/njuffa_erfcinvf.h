/**
njuffa_erfcinvf.h
====================

https://stackoverflow.com/questions/60472139/computing-the-inverse-of-the-complementary-error-function-erfcinv-in-c

This C CPU implementation by njuffa (from stackoverflow) of the inverse of the complementary 
error function gives results very close to those from the CUDA erfcinvf function.

**/


#include <math.h>
#include <stdint.h>

#define PORTABLE  (1)

float njuffa_logf (float a);
#if !PORTABLE
#include "immintrin.h"
float sse_recipf (float a);
float sse_rsqrtf (float a);
#endif // !PORTABLE

/* Compute inverse of the complementary error function. For the central region,
   re-use the polynomial approximation for erfinv. For the tail regions, use an
   approximation based on the observation that erfcinv(x) is very approximately
   sqrt(-log(x)).

   PORTABLE=1 max. ulp err. = 3.12017
   PORTABLE=0 max. ulp err. = 3.13523
*/
float njuffa_erfcinvf (float a)
{
    float r;

    if ((a >= 2.1875e-3f) && (a <= 1.998125f)) { // max. ulp err. = 2.77667
        float p, t;
        t = fmaf (-a, a, a + a);
        t = njuffa_logf (t);
        p =              5.43877832e-9f;  //  0x1.75c000p-28 
        p = fmaf (p, t,  1.43286059e-7f); //  0x1.33b458p-23 
        p = fmaf (p, t,  1.22775396e-6f); //  0x1.49929cp-20 
        p = fmaf (p, t,  1.12962631e-7f); //  0x1.e52bbap-24 
        p = fmaf (p, t, -5.61531961e-5f); // -0x1.d70c12p-15 
        p = fmaf (p, t, -1.47697705e-4f); // -0x1.35be9ap-13 
        p = fmaf (p, t,  2.31468701e-3f); //  0x1.2f6402p-9 
        p = fmaf (p, t,  1.15392562e-2f); //  0x1.7a1e4cp-7 
        p = fmaf (p, t, -2.32015476e-1f); // -0x1.db2aeep-3 
        t = fmaf (p, t,  8.86226892e-1f); //  0x1.c5bf88p-1 
        r = fmaf (t, -a, t);
    } else {
        float p, q, s, t;
        t = (a >= 1.0f) ? (2.0f - a) : a;
        t = 0.0f - njuffa_logf (t);
#if PORTABLE
        s = sqrtf (1.0f / t);
#else // PORTABLE
        s = sse_rsqrtf (t);
#endif // PORTABLE
        p =              2.23100796e+1f;  //  0x1.64f616p+4
        p = fmaf (p, s, -5.23008537e+1f); // -0x1.a26826p+5
        p = fmaf (p, s,  5.44409714e+1f); //  0x1.b3871cp+5
        p = fmaf (p, s, -3.35030403e+1f); // -0x1.0c063ap+5
        p = fmaf (p, s,  1.38580027e+1f); //  0x1.bb74c2p+3
        p = fmaf (p, s, -4.37277269e+0f); // -0x1.17db82p+2
        p = fmaf (p, s,  1.53075826e+0f); //  0x1.87dfc6p+0
        p = fmaf (p, s,  2.97993328e-2f); //  0x1.e83b76p-6
        p = fmaf (p, s, -3.71997419e-4f); // -0x1.86114cp-12
        p = fmaf (p, s, s);
#if PORTABLE
        r = 1.0f / p;
#else // PORTABLE
        r = sse_recipf (p);
        if (t == INFINITY) r = t;
#endif // PORTABLE
        if (a >= 1.0f) r = 0.0f - r;
    }
    return r;
}

/* Compute inverse of the CDF of the standard normal distribution.
   max ulp err = 4.08385
*/
float njuffa_normcdfinvf (float a)
{
    return fmaf (-1.41421356f, njuffa_erfcinvf (a + a), 0.0f);
}

/* natural logarithm. max ulp err = 0.85089 */
float njuffa_logf (float a)
{
    float m, r, s, t, i, f;
    int32_t e;
    const float cutoff_f = 0.666666667f;
    if ((a > 0.0f) && (a <=  0x1.fffffep+127f)) { // 3.40282347e+38
        m = frexpf (a, &e);
        if (m < cutoff_f) {
            m = m + m;
            e = e - 1;
        }
        i = (float)e;
        f = m - 1.0f;
        s = f * f;
        /* Compute log1p(f) for f in [-1/3, 1/3] */
        r =             -0x1.0ae000p-3f;  // -0.130310059
        t =              0x1.208000p-3f;  //  0.140869141
        r = fmaf (r, s, -0x1.f1988ap-4f); // -0.121483363
        t = fmaf (t, s,  0x1.1e5740p-3f); //  0.139814854
        r = fmaf (r, s, -0x1.55b36ep-3f); // -0.166846141
        t = fmaf (t, s,  0x1.99d8b2p-3f); //  0.200120345
        r = fmaf (r, s, -0x1.fffe02p-3f); // -0.249996200
        r = fmaf (t, f, r);
        r = fmaf (r, f,  0x1.5554fap-2f); //  0.333331972
        r = fmaf (r, f, -0x1.000000p-1f); // -0.500000000
        r = fmaf (r, s, f);
        r = fmaf (i, 0x1.62e430p-01f, r); //  0.693147182 // log(2)
    } else {
        r = a + a;  // silence NaNs if necessary
        if (a  < 0.0f) r = 0.0f/0.0f; // QNaN INDEFINITE
        if (a == 0.0f) r = -INFINITY; // -INF
    }
    return r;
}

#if !PORTABLE
float sse_recipf (float a)
{
    __m128 t;
    float e, r;
    t = _mm_set_ss (a);
    t = _mm_rcp_ss (t);
    _mm_store_ss (&r, t);
    e = fmaf (0.0f - a, r, 1.0f);
    e = fmaf (e, e, e);
    r = fmaf (e, r, r);
    return r;
}


float sse_rsqrtf (float a)
{
    __m128 t;
    float e, r;
    t = _mm_set_ss (a);
    t = _mm_rsqrt_ss (t);
    _mm_store_ss (&r, t);
    e = fmaf (0.0f - a, r * r, 1.0f);
    r = fmaf (fmaf (0.375f, e, 0.5f), e * r, r);
    return r;
}
#endif

