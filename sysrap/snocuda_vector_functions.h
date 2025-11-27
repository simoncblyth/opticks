#pragma once

struct float4 { float x,y,z,w ; };
struct float3 { float x,y,z   ; };
struct float2 { float x,y ; };

struct double4 { double x,y,z,w ; };
struct double3 { double x,y,z   ; };
struct double2 { double x,y ; };

struct int4 { int x,y,z,w ; };
struct int3 { int x,y,z   ; };
struct int2 { int x,y     ; };
struct int1 { int x       ; };

struct uint4  { unsigned x,y,z,w ; };
struct uint3  { unsigned x,y,z   ; };
struct uint2  { unsigned x,y ; };
struct uint1  { unsigned x ; };

struct char4   { char x,y,z,w ; };
struct uchar4  { unsigned char x,y,z,w ; };
struct short4  { short x,y,z,w ; };
struct ushort4 { unsigned short x,y,z,w ; };

struct longlong4 { long long x, y, z, w ; };
struct longlong3 { long long x, y, z ; };
struct longlong2 { long long x, y ; };
struct longlong1 { long long x ; };

struct ulonglong4 { unsigned long long x, y, z, w ; };
struct ulonglong3 { unsigned long long x, y, z ; };
struct ulonglong2 { unsigned long long x, y ; };
struct ulonglong1 { unsigned long long x ; };



inline double4 make_double4(double x, double y, double z, double w){ double4 v{x,y,z,w} ; return v ; }
inline double3 make_double3(double x, double y, double z){           double3 v{x,y,z} ; return v ; }
inline double2 make_double2(double x, double y){                     double2 v{x,y} ; return v ; }



inline float4 make_float4(float x, float y, float z, float w){ float4 v{x,y,z,w} ; return v ; }
inline float3 make_float3(float x, float y, float z){          float3 v{x,y,z} ; return v ; }
inline float2 make_float2(float x, float y){                   float2 v{x,y} ; return v ; }

inline int2 make_int2(int x, int y){  int2 v{x,y} ; return v ; }
inline int3 make_int3(int x, int y, int z){  int3 v{x,y,z} ; return v ; }
inline int4 make_int4(int x, int y, int z, int w){  int4 v{x,y,z,w} ; return v ; }

inline uint2 make_uint2(unsigned x, unsigned y){  uint2 v{x,y} ; return v ; }
inline uint3 make_uint3(unsigned x, unsigned y, unsigned z){  uint3 v{x,y,z} ; return v ; }
inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w){  uint4 v{x,y,z,w} ; return v ; }

inline longlong4 make_longlong4(long long x, long long y, long long z, long long w){ longlong4 v{x,y,z,w} ; return v ; }
inline longlong3 make_longlong3(long long x, long long y, long long z){ longlong3 v{x,y,z} ; return v ; }
inline longlong2 make_longlong2(long long x, long long y){ longlong2 v{x,y} ; return v ; }

inline ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w){ ulonglong4 v{x,y,z,w} ; return v ; }
inline ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z){ ulonglong3 v{x,y,z} ; return v ; }
inline ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y){ ulonglong2 v{x,y} ; return v ; }



