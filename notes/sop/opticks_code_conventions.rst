opticks_code_conventions
==========================


CMake naming, structure and layout
-----------------------------------

1. project names are the same as the library targets they generate 
2. project names are CamelCased, project directory names are lowercase
3. lowercased project names match the names of their directories
4. every CMake project command should be followed by "include(OpticksBuildOptions)"
5. the name of every project should be set via a "name" variable : this is used within
   the OpticksBuildOptions to set the install location of headers as include/${name}

For example a project called DemoProj would be setup in demoproj subdirectory with
a CMakeLists.txt file starting::

    cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
    set(name DemoProj)
    project(${name} VERSION 0.1.0)
    include(OpticksBuildOptions)


Integrated build with a top level CMakeLists.txt OR NOT, Currently NOT
----------------------------------------------------------------------

Because it doesnt add enough value for the effort have halted
the attempt to setup a top level CMakeLists.txt : instead 
just use proj-by-proj integrated at bash level with om.bash, 
see om-install

The big advantages of independent packages:

1. fast development builds
2. independent packages really are independent - with different dependencies

   * this proved vital for the server/client split builds


Integrated Build  : ON HOLD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One approach to providing an integrated CMake level build as well
as separate ones is to redefine find_package to skip packages that have directly 
been built already within the single CMake context and hence do not need to be found.

This structure allows the proj-by-proj building and integrated building to otherwise
be identical.

bcm_ignore_package
~~~~~~~~~~~~~~~~~~~

Incase decide to try again with integrated build, try 
out the bcm_ignore_package approach:

* http://bcm.readthedocs.io/en/latest/src/Building.html#integrated-builds





