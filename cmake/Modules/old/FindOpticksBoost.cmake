# Somewhat awkward setting of global properties avoids repeatedly and verbosely finding boost.
#
# Policy push/pop is to suppress warnings from non-existing targets used in Boost cmake machinery 
#
#     cmake --help-policy CMP0045
#     vi /usr/lib64/boost/BoostConfig.cmake +72 
#
# use the cmake supplied FindBoost.cmake not BoostConfig.cmake
set(Boost_NO_BOOST_CMAKE ON)


set(OPTICKS_BOOST_COMPONENTS
    system 
    program_options 
    filesystem 
    regex 
)

# try without : thread 
# no longer using: log log_setup


# use of timer is being evaluated on mac in brap-/tests/BTimeTest
#if(APPLE)
#   list(APPEND OPTICKS_BOOST_COMPONENTS thread timer)
#endif(APPLE)



get_property(OpticksBoost_FOUND        GLOBAL PROPERTY gOpticksBoost_FOUND SET)

if(OpticksBoost_FOUND)
    get_property(OpticksBoost_LIBRARIES    GLOBAL PROPERTY gOpticksBoost_LIBRARIES)
    get_property(OpticksBoost_INCLUDE_DIRS GLOBAL PROPERTY gOpticksBoost_INCLUDE_DIRS)
    get_property(OpticksBoost_DEFINITIONS  GLOBAL PROPERTY gOpticksBoost_DEFINITIONS)
    #message("Already found OpticksBoost : ${OpticksBoost_LIBRARIES}")
else()
    #message("looking for OpticksBoost")
    if (CMAKE_VERSION VERSION_EQUAL "3.0" OR CMAKE_VERSION VERSION_GREATER "3.0")
        cmake_policy(PUSH)
        cmake_policy(SET CMP0045 OLD)
    endif()


    #set(Boost_DEBUG 1) 

    find_package(Boost REQUIRED COMPONENTS ${OPTICKS_BOOST_COMPONENTS} )

    if (CMAKE_VERSION VERSION_EQUAL "3.0" OR CMAKE_VERSION VERSION_GREATER "3.0")
        cmake_policy(POP)
    endif()

    if(Boost_FOUND)
        set_property(GLOBAL PROPERTY gOpticksBoost_FOUND "YES")
        set_property(GLOBAL PROPERTY gOpticksBoost_LIBRARIES    ${Boost_LIBRARIES})
        set_property(GLOBAL PROPERTY gOpticksBoost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})

        if(APPLE)
            set_property(GLOBAL PROPERTY gOpticksBoost_DEFINITIONS  ${Boost_DEFINITIONS} -DBOOST_LOG_DYN_LINK)
        endif(APPLE)
    endif(Boost_FOUND)

endif()



