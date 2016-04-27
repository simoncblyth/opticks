
# use the cmake supplied FindBoost.cmake not BoostConfig.cmake
set(Boost_NO_BOOST_CMAKE ON)

if (CMAKE_VERSION VERSION_EQUAL "3.0" OR CMAKE_VERSION VERSION_GREATER "3.0")

    # Suppress warnings from non-existing targets used in Boost cmake machinery 
    #
    #     cmake --help-policy CMP0045
    #     vi /usr/lib64/boost/BoostConfig.cmake +72 
    #
    cmake_policy(PUSH)
    cmake_policy(SET CMP0045 OLD)
endif()

find_package(Boost REQUIRED COMPONENTS system thread program_options log log_setup filesystem regex)


if (CMAKE_VERSION VERSION_EQUAL "3.0" OR CMAKE_VERSION VERSION_GREATER "3.0")
    cmake_policy(POP)
endif()


if(Boost_FOUND)
set(OpticksBoost_FOUND ON)
set(OpticksBoost_LIBRARIES    ${Boost_LIBRARIES})
set(OpticksBoost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
set(OpticksBoost_DEFINITIONS  ${Boost_DEFINITIONS} -DBOOST_LOG_DYN_LINK)
endif(Boost_FOUND)


