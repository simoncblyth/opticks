set(ImplicitMesher_PREFIX "${CMAKE_INSTALL_PREFIX}/externals")

find_path( ImplicitMesher_INCLUDE_DIR 
           NAMES ImplicitMesher/ImplicitMesherF.h
           PATHS ${ImplicitMesher_PREFIX}/include
)

find_library( ImplicitMesher_LIBRARY
              NAMES ImplicitMesher
              PATHS ${ImplicitMesher_PREFIX}/lib )


if(ImplicitMesher_INCLUDE_DIR AND ImplicitMesher_LIBRARY)
   set(ImplicitMesher_FOUND "YES")
else()
   set(ImplicitMesher_FOUND "NO")
endif()


include(EchoTarget)
echo_pfx_vars(ImplicitMesher "FOUND;INCLUDE_DIR;LIBRARY;PREFIX")


if(ImplicitMesher_FOUND AND NOT TARGET Opticks::ImplicitMesher)
    add_library(Opticks::ImplicitMesher UNKNOWN IMPORTED) 
    set_target_properties(Opticks::ImplicitMesher PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ImplicitMesher_INCLUDE_DIR}"
        IMPORTED_LOCATION "${ImplicitMesher_LIBRARY}"
    )
endif()


# hmm ImplicitMesher is a CMake built project under my control
# so could avoid this Find glue and export the targets directly 
# from the project ?
#

