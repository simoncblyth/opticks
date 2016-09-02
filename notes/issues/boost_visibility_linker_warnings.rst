Boost Visibility Linker Warnings
==================================

Issue : many linker warnings
-------------------------------

Most opticks pkgs have many warnings like::

    ld: warning: direct access in void boost::throw_exception<boost::bad_function_call>(boost::bad_function_call const&) to global weak symbol typeinfo for boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::bad_function_call> > means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::bad_function_call> >::rethrow() const to global weak symbol typeinfo for boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::bad_function_call> > means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.
    ld: warning: direct access in void boost::throw_exception<boost::bad_lexical_cast>(boost::bad_lexical_cast const&) to global weak symbol typeinfo for boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::bad_lexical_cast> > means the weak symbol cannot be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings.


Solution ?
-----------

Rebuild boost libs with::

   -fvisibility=hidden

* http://stackoverflow.com/questions/15059360/compiling-boost-1-53-libraries-with-gcc-with-symbol-visibility-hidden


Cause
--------

* On Linux/OSX are using **visibility=hidden** in order to match WIN32 behaviour.
* This forces all symbols to be API exported eg with **OKG4_API**

okg4/OKG4_API_EXPORT.hh::

     23 #if defined (_WIN32) 
     24 
     25    #if defined(okg4_EXPORTS)
     26        #define  OKG4_API __declspec(dllexport)
     27    #else
     28        #define  OKG4_API __declspec(dllimport)
     29    #endif
     30 
     31 #else
     32 
     33    #define OKG4_API  __attribute__ ((visibility ("default")))
     34 
     35 #endif


cmake/Modules/EnvCompilationFlags::

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")


