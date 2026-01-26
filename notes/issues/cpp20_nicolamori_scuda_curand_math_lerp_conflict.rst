cpp20_nicolamori_scuda_curand_math_lerp_conflict
=================================================


issue reported by nicolamori
-----------------------------

I am using the eic-opticks variant but I think this problem applies also to the
original opticks. When compiling with C++20 I get the following error::

    [ 98%] Building CXX object src/CMakeFiles/gphox.dir/config.cpp.o
    In file included from /home/mori/software/KM3/eic-opticks/sysrap/sphoton.h:183,
                     from /home/mori/software/KM3/eic-opticks/sysrap/storch.h:37,
                     from /home/mori/software/KM3/eic-opticks/src/config.h:8,
                     from /home/mori/software/KM3/eic-opticks/src/config.cpp:14:
    /home/mori/software/KM3/eic-opticks/sysrap/scuda.h:130:85: error: 'float lerp(float, float, float)' conflicts with a previous declaration
      130 | SUTIL_INLINE SUTIL_HOSTDEVICE float lerp(const float a, const float b, const float t)
          |                                                                                     ^
    In file included from /usr/include/c++/15.2.1/valarray:41,
                     from /home/mori/software/install/cpp20/NLOHMANNJSON_v3.11.3/include/nlohmann/detail/conversions/from_json.hpp:21,
                     from /home/mori/software/install/cpp20/NLOHMANNJSON_v3.11.3/include/nlohmann/adl_serializer.hpp:14,
                     from /home/mori/software/install/cpp20/NLOHMANNJSON_v3.11.3/include/nlohmann/json.hpp:34,
                     from /home/mori/software/KM3/eic-opticks/src/config.cpp:9:
    /usr/include/c++/15.2.1/cmath:3857:3: note: previous declaration 'constexpr float std::lerp(float, float, float)'
     3857 |   lerp(float __a, float __b, float __t) noexcept
          |   ^~~~


C++20 defines lerp in <cmath> under the namespace std; this causes no problem,
but it turns out that the included curand_kernel.h includes the C header math.h
which in turn declare using std::lerp;. This moves std::lerp in the global
namespace and causes a conflict with the lerp definition at line 149 of
sysrap/scuda.h.

A possible fix which seems to work (but more testing is needed) is::

    diff --git a/sysrap/scuda.h b/sysrap/scuda.h
    index c36293339..26eb11bca 100644
    --- a/sysrap/scuda.h
    +++ b/sysrap/scuda.h
    @@ -125,13 +125,15 @@ SUTIL_INLINE SUTIL_HOSTDEVICE unsigned long long min(unsigned long long a, unsig
         return a < b ? a : b;
     }
     
    -
    -/** lerp */
    +#if __cplusplus <= 201703L
    +/** lerp **/
     SUTIL_INLINE SUTIL_HOSTDEVICE float lerp(const float a, const float b, const float t)
     {
       return a + t*(b-a);
     }
    -
    +#else
    +using std::lerp;
    +#endif


The only downside I see is that this makes the SUTIL_INLINE SUTIL_HOSTDEVICE
macros not used with C++20. However, the code section in sysrap/scuda.h where
lerp is defined is guarded by #if !defined(__CUDACC__) so that the neglected
macros would simply be equivaent to inline and thus I guess no side effect can
come from the patch.

@simoncblyth what do you think about this?


attempt to reproduce didnt do so
------------------------------------

::

    CPPSTD=c++20 ~/o/sysrap/tests/sphoton_test.sh build




