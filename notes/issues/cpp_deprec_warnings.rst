cpp_deprec_warnings
======================

Hans reports lots of warnings with his update expts.



wenzel@irago:/data3/wenzel/newopticks_dev2$ grep C++11 install_full.log
/usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:131:13: warning: Thrust requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
  131 |      THRUST_COMPILER_DEPRECATION_SOFT(C++14, C++11);
/usr/local/cuda/include/cub/util_cpp_dialect.cuh:142:13: warning: CUB requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define CUB_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
  142 |      CUB_COMPILER_DEPRECATION_SOFT(C++14, C++11);
/usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:131:13: warning: Thrust requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
  131 |      THRUST_COMPILER_DEPRECATION_SOFT(C++14, C++11);
/usr/local/cuda/include/cub/util_cpp_dialect.cuh:142:13: warning: CUB requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define CUB_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
  142 |      CUB_COMPILER_DEPRECATION_SOFT(C++14, C++11);

wenzel@irago:/data3/wenzel/newopticks_dev2$ grep warning  install_full.log
/usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:131:13: warning: Thrust requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
/usr/local/cuda/include/cub/util_cpp_dialect.cuh:142:13: warning: CUB requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define CUB_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
/usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:131:13: warning: Thrust requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
/usr/local/cuda/include/cub/util_cpp_dialect.cuh:142:13: warning: CUB requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define CUB_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
/usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:131:13: warning: Thrust requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
/usr/local/cuda/include/cub/util_cpp_dialect.cuh:142:13: warning: CUB requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define CUB_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
/usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:131:13: warning: Thrust requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
/usr/local/cuda/include/cub/util_cpp_dialect.cuh:142:13: warning: CUB requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define CUB_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.


/data3/wenzel/newopticks_dev2/opticks/sysrap/SDigest.cc:191:14: warning: ‘int MD5_Init(MD5_CTX*)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/SDigest.cc:193:20: warning: ‘int MD5_Update(MD5_CTX*, const void*, size_t)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/SDigest.cc:199:15: warning: ‘int MD5_Final(unsigned char*, MD5_CTX*)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/SDigest.cc:215:13: warning: ‘int MD5_Init(MD5_CTX*)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/SDigest.cc:221:23: warning: ‘int MD5_Update(MD5_CTX*, const void*, size_t)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/SDigest.cc:223:23: warning: ‘int MD5_Update(MD5_CTX*, const void*, size_t)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/SDigest.cc:230:14: warning: ‘int MD5_Final(unsigned char*, MD5_CTX*)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/SDigest.cc:243:13: warning: ‘int MD5_Init(MD5_CTX*)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/SDigest.cc:249:23: warning: ‘int MD5_Update(MD5_CTX*, const void*, size_t)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/SDigest.cc:251:23: warning: ‘int MD5_Update(MD5_CTX*, const void*, size_t)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/SDigest.cc:258:14: warning: ‘int MD5_Final(unsigned char*, MD5_CTX*)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/SDigest.cc:275:23: warning: ‘int MD5_Update(MD5_CTX*, const void*, size_t)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/SDigest.cc:277:23: warning: ‘int MD5_Update(MD5_CTX*, const void*, size_t)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/SDigest.cc:287:14: warning: ‘int MD5_Final(unsigned char*, MD5_CTX*)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/SDigest.cc:301:13: warning: ‘int MD5_Init(MD5_CTX*)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/SDigest.cc:326:12: warning: ‘int MD5_Init(MD5_CTX*)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]


/data3/wenzel/newopticks_dev2/opticks/sysrap/sdigest.h:55:36: warning: ‘int MD5_Init(MD5_CTX*)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/sdigest.h:81:13: warning: ‘int MD5_Init(MD5_CTX*)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/sdigest.h:89:13: warning: ‘int MD5_Init(MD5_CTX*)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/sdigest.h:142:23: warning: ‘int MD5_Update(MD5_CTX*, const void*, size_t)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/sdigest.h:144:23: warning: ‘int MD5_Update(MD5_CTX*, const void*, size_t)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]
/data3/wenzel/newopticks_dev2/opticks/sysrap/sdigest.h:154:14: warning: ‘int MD5_Final(unsigned char*, MD5_CTX*)’ is deprecated: Since OpenSSL 3.0 [-Wdeprecated-declarations]


/usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:131:13: warning: Thrust requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
/usr/local/cuda/include/cub/util_cpp_dialect.cuh:142:13: warning: CUB requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define CUB_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
/usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:131:13: warning: Thrust requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
/usr/local/cuda/include/cub/util_cpp_dialect.cuh:142:13: warning: CUB requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define CUB_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
/usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:131:13: warning: Thrust requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
/usr/local/cuda/include/cub/util_cpp_dialect.cuh:142:13: warning: CUB requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define CUB_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
/usr/local/cuda/include/thrust/detail/config/cpp_dialect.h:131:13: warning: Thrust requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
/usr/local/cuda/include/cub/util_cpp_dialect.cuh:142:13: warning: CUB requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define CUB_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.




Dealt with these::

    /data3/wenzel/newopticks_dev2/opticks/ggeo/GTransforms.cc:68:8: warning: ‘identity’ may be used uninitialized [-Wmaybe-uninitialized]

    /data3/wenzel/newopticks_dev2/opticks/extg4/tests/X4MaterialPropertiesTableTest.cc:92:10: warning: unused variable ‘all’ [-Wunused-variable]

    /data3/wenzel/newopticks_dev2/opticks/u4/U4Material.cc:362:32: warning: unused variable ‘mpt’ [-Wunused-variable]

    /data3/wenzel/newopticks_dev2/opticks/u4/U4Process.h:175:21: warning: comparison of integer expressions of different signedness: ‘int’ and ‘std::size_t’ {aka ‘long unsigned int’} [-Wsign-compare]



    /data3/wenzel/newopticks_dev2/opticks/CSGOptiX/Check.cu(11): warning #1444-D: function "float_as_uint"
    /usr/local/cuda/include/crt/device_functions.hpp(140): here was declared deprecated ("float_as_uint() is deprecated in favor of __float_as_uint() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning).")

    /data3/wenzel/newopticks_dev2/opticks/CSGOptiX/Check.cu(12): warning #1444-D: function "uint_as_float"
    /usr/local/cuda/include/crt/device_functions.hpp(145): here was declared deprecated ("uint_as_float() is deprecated in favor of __uint_as_float() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning).")

    /data3/wenzel/newopticks_dev2/opticks/CSGOptiX/CSGOptiX7.cu(171): warning #1444-D: function "uint_as_float"
    /usr/local/cuda/include/crt/device_functions.hpp(145): here was declared deprecated ("uint_as_float() is deprecated in favor of __uint_as_float() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning).")



