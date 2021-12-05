cmake_3_21_2_new_warnings_vs_3_17_1
======================================



New warning in okop : simply due to omitted extensions
--------------------------------------------------------

::

    -- OpticksCompilationFlags.cmake : CMAKE_CXX_STANDARD_REQUIRED : on 
    -- Configuring done
    CMake Warning (dev) at tests/CMakeLists.txt:17 (add_executable):
      Policy CMP0115 is not set: Source file extensions must be explicit.  Run
      "cmake --help-policy CMP0115" for policy details.  Use the cmake_policy
      command to set the policy and suppress this warning.

      File:

        /home/blyth/opticks/okop/tests/dirtyBufferTest.cc
    This warning is for project developers.  Use -Wno-dev to suppress it.

    CMake Warning (dev) at tests/CMakeLists.txt:17 (add_executable):
      Policy CMP0115 is not set: Source file extensions must be explicit.  Run
      "cmake --help-policy CMP0115" for policy details.  Use the cmake_policy
      command to set the policy and suppress this warning.

      File:

        /home/blyth/opticks/okop/tests/compactionTest.cc
    This warning is for project developers.  Use -Wno-dev to suppress it.

    -- Generating done
    -- Build files have been written to: /data/blyth/junotop/ExternalLibs/opticks/head/build/okop
    === om-make-one : okop            /home/blyth/opticks/okop                                     /data/blyth/junotop/ExternalLibs/opticks/head/build/okop     
    [  4%] Building NVCC (Device) object CMakeFiles/OKOP.dir/OKOP_genera



CMP0115
---------


::

    O[blyth@localhost sysrap]$ cmake --help-policy CMP0115
    CMP0115
    -------

    Source file extensions must be explicit.

    In CMake 3.19 and below, if a source file could not be found by the name
    specified, it would append a list of known extensions to the name to see if
    the file with the extension could be found. For example, this would allow the
    user to run:

     add_executable(exe main)

    and put ``main.c`` in the executable without specifying the extension.

    Starting in CMake 3.20, CMake prefers all source files to have their extensions
    explicitly listed:

     add_executable(exe main.c)

    The ``OLD`` behavior for this policy is to implicitly append known extensions
    to source files if they can't be found. The ``NEW`` behavior of this policy is
    to not append known extensions and require them to be explicit.

    This policy was introduced in CMake version 3.20.  CMake version 3.21.2
    warns when the policy is not set and uses ``OLD`` behavior. Use the
    ``cmake_policy()`` command to set it to ``OLD`` or ``NEW`` explicitly.

    .. note::
      The ``OLD`` behavior of a policy is
      ``deprecated by definition``
      and may be removed in a future version of CMake.
    O[blyth@localhost sysrap]$ 




All CUDA using proj are giving CMake warnings
--------------------------------------------------

::

    [100%] Built target QSimTest
    -- Configuring QUDARap
    CMake Deprecation Warning at CMakeLists.txt:7 (cmake_policy):
      The OLD behavior for policy CMP0077 will be removed from a future version
      of CMake.

      The cmake-policies(7) manual explains that the OLD behaviors of all
      policies are deprecated and that a policy should be set to OLD only under
      specific short-term circumstances.  Projects should be ported to the NEW
      behavior and not rely on setting a policy to OLD.


    -- FindOpticksCUDA.cmake:with-policy-CMP0077
    CMake Deprecation Warning at /home/blyth/opticks/cmake/Modules/FindOpticksCUDA.cmake:33 (cmake_policy):
      The OLD behavior for policy CMP0077 will be removed from a future version
      of CMake.

      The cmake-policies(7) manual explains that the OLD behaviors of all
      policies are deprecated and that a policy should be set to OLD only under
      specific short-term circumstances.  Projects should be ported to the NEW
      behavior and not rely on setting a policy to OLD.
    Call Stack (most recent call first):
      CMakeLists.txt:18 (find_package)




::

    O[blyth@localhost opticks]$ which cmake
    /data/blyth/junotop/ExternalLibs/Cmake/3.21.2/bin/cmake

    O[blyth@localhost opticks]$ cmake --version
    cmake version 3.21.2

    CMake suite maintained and supported by Kitware (kitware.com/cmake).
    O[blyth@localhost opticks]$ cmake --help-policy CMP0077
    CMP0077
    -------

    ``option()`` honors normal variables.

    The ``option()`` command is typically used to create a cache entry
    to allow users to set the option.  However, there are cases in which a
    normal (non-cached) variable of the same name as the option may be
    defined by the project prior to calling the ``option()`` command.
    For example, a project that embeds another project as a subdirectory
    may want to hard-code options of the subproject to build the way it needs.

    For historical reasons in CMake 3.12 and below the ``option()``
    command *removes* a normal (non-cached) variable of the same name when:

    * a cache entry of the specified name does not exist at all, or
    * a cache entry of the specified name exists but has not been given
      a type (e.g. via ``-D<name>=ON`` on the command line).

    In both of these cases (typically on the first run in a new build tree),
    the ``option()`` command gives the cache entry type ``BOOL`` and
    removes any normal (non-cached) variable of the same name.  In the
    remaining case that the cache entry of the specified name already
    exists and has a type (typically on later runs in a build tree), the
    ``option()`` command changes nothing and any normal variable of
    the same name remains set.

    In CMake 3.13 and above the ``option()`` command prefers to
    do nothing when a normal variable of the given name already exists.
    It does not create or update a cache entry or remove the normal variable.
    The new behavior is consistent between the first and later runs in a
    build tree.  This policy provides compatibility with projects that have
    not been updated to expect the new behavior.

    When the ``option()`` command sees a normal variable of the given
    name:

    * The ``OLD`` behavior for this policy is to proceed even when a normal
      variable of the same name exists.  If the cache entry does not already
      exist and have a type then it is created and/or given a type and the
      normal variable is removed.

    * The ``NEW`` behavior for this policy is to do nothing when a normal
      variable of the same name exists.  The normal variable is not removed.
      The cache entry is not created or updated and is ignored if it exists.

    See ``CMP0126`` for a similar policy for the ``set(CACHE)``
    command, but note that there are some differences in ``NEW`` behavior
    between the two policies.

    This policy was introduced in CMake version 3.13.  CMake version
    3.21.2 warns when the policy is not set and uses ``OLD`` behavior.
    Use the ``cmake_policy()`` command to set it to ``OLD`` or ``NEW``
    explicitly.

    .. note::
      The ``OLD`` behavior of a policy is
      ``deprecated by definition``
      and may be removed in a future version of CMake.
    O[blyth@localhost opticks]$ 




man cmake-policies
----------------------

::

    Policies in CMake are used to preserve backward compatible behavior
    across multiple releases.  When a new policy is introduced, newer CMake
    versions will begin to warn about the backward compatible behavior.  It is
    possible to disable the warning by explicitly requesting the OLD, or backward
    compatible behavior using the cmake_policy() command.  It is  also  possible
    to request  NEW,  or non-backward compatible behavior for a policy, also
    avoiding the warning.  Each policy can also be set to either NEW or OLD
    behavior explicitly on the command line with the CMAKE_POLICY_DEFAULT_CMP<NNNN>
    variable.

    A policy is a deprecation mechanism and not a reliable feature toggle.
    A policy should almost never be set to OLD, except to silence warnings in an
    otherwise frozen or stable  codebase, or temporarily as part of a larger
    migration path. The OLD behavior of each policy is undesirable and will be
    replaced with an error condition in a future release.

    The  cmake_minimum_required() command does more than report an error if
    a too-old version of CMake is used to build a project.  It also sets all
    policies introduced in that CMake version or earlier to NEW behavior.  To
    manage policies without increasing the minimum required CMake version, the
    if(POLICY) command may be used:

          if(POLICY CMP0990)
            cmake_policy(SET CMP0990 NEW)
          endif()

    This has the effect of using the NEW behavior with newer CMake releases
    which users may be using and not issuing a compatibility warning.

    The setting of a policy is confined in some cases to not propagate to
    the parent scope.  For example, if the files read by the include() command or
    the find_package() command  contain  a use of cmake_policy(), that policy
    setting will not affect the caller by default.  Both commands accept an
    optional NO_POLICY_SCOPE keyword to control this behavior.

    The CMAKE_MINIMUM_REQUIRED_VERSION variable may also be used to
    determine whether to report an error on use of deprecated macros or functions.




Getting warnings from use of OLD policy
-------------------------------------------

Putting policy into OpticksBuildOpticks.cmake and including
that with NO_POLICY_SCOPE might avoid duplication::

    include(OpticksBuildOptions NO_POLICY_SCOPE) 


::

    O[blyth@localhost opticks]$ find . -name CMakeLists.txt -exec grep -H OLD {} \;
    ./CMakeLists.txt:This Integrated Build  : ON HOLD
    ./cfg4/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./cudarap/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./g4ok/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./ok/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./okg4/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./okop/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./opticksgl/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./optixrap/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./qudarap/CMakeLists.txt:#    cmake_policy(SET CMP0077 OLD)
    ./thrustrap/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)

    ./examples/ThrustOpenGLInterop/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseCFG4/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseG4OK/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOContextBufferPP/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOKG4/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOKOP/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXBuffer/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXBufferPP/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXFan/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXGeometry/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXGeometryInstanced/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXGeometryInstancedOCtx/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXGeometryInstancedStandalone/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXGeometryOCtx/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXGeometryStandalone/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXGeometryTriangles/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXProgram/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXProgramPP/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXRap/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXTexture/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXTextureLayered/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXTextureLayeredOK/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXTextureLayeredOKImg/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXTextureLayeredOKImgGeo/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOptiXTextureLayeredPP/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOpticksCUDA/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UseOpticksGL/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)
    ./examples/UsePLogChained/CMakeLists.txt:#set_target_properties(ChainedApp PROPERTIES FOLDER Samples/Chained)
    ./examples/UsePLogChained/CMakeLists.txt:#set_target_properties(ChainedLib PROPERTIES FOLDER Samples/Chained)
    ./examples/UseThrustRap/CMakeLists.txt:    cmake_policy(SET CMP0077 OLD)


    O[blyth@localhost opticks]$ vi 


