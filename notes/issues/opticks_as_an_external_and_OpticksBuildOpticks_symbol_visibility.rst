opticks_as_an_external_and_OpticksBuildOpticks_symbol_visibility
==================================================================


::

    Hi Hans, Soon, 

    Do I understand correctly that for convenience 
    of using Opticks as an external you propose two changes
    to cmake/Modules/OpticksBuildOptions.cmake::

    1. dont set CMAKE_INSTALL_RPATH
    2. dont set CMAKE_CXX_FLAGS by not doing "include(OpticksCXXFlags)"

    The Opticks packages need hidden symbol visibility from::

      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")

    I would expect not having hidden symbols would cause many issues,
    did you compile Opticks like that ? If so did logging work ? 

    Originally I started using hidden visibility when I ported an ancient 
    version to Windows. Also I recall logging issues when trying to work 
    without hidden symbols : you cannot have distinct plog 
    loggers for each library in that case.

    Simon



About Symbol Visibility
-------------------------

* https://gcc.gnu.org/wiki/Visibility

* https://labjack.com/blogs/news/simple-c-symbol-visibility-demo

* https://gist.github.com/ax3l/ba17f4bb1edb5885a6bd01f58de4d542

  Visible Symbols in C++ Projects


CMake and symbol visibility
-----------------------------

* https://stackoverflow.com/questions/17080869/what-is-the-cmake-equivalent-to-gcc-fvisibility-hidden-when-controlling-the-e


Instead of setting compiler flags directly, you should be using a current CMake
version and the <LANG>_VISIBILITY_PRESET properties instead. This way you can
avoid compiler specifics in your CMakeLists and improve cross platform
applicability (avoiding errors such as supporting GCC and not Clang).

I.e., if you are using C++ you would either call
set(CMAKE_CXX_VISIBILITY_PRESET hidden) to set the property globally, or
set_target_properties(MyTarget PROPERTIES CXX_VISIBILITY_PRESET hidden) to
limit the setting to a specific library or executable target. If you are using
C just replace CXX by C in the aforementioned commands. You may also want to
investigate the VISIBLITY_INLINES_HIDDEN property as well.

The documentation for GENERATE_EXPORT_HEADER includes some more tips and
examples related to both properties.

* https://cmake.org/cmake/help/latest/prop_tgt/VISIBILITY_INLINES_HIDDEN.html
* https://cmake.org/cmake/help/latest/prop_tgt/LANG_VISIBILITY_PRESET.html#prop_tgt:%3CLANG%3E_VISIBILITY_PRESET
* https://cmake.org/cmake/help/latest/module/GenerateExportHeader.html













