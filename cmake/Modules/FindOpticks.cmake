#[=[
FindOpticks.cmake
===================

Hmm: could keep the G4OK internal/implicit for simplicity as do not need 
the flexibility to access other "COMPONENTS" currently. 

Testing this from ~/opticks/examples/UseOpticks/go.sh 


https://stackoverflow.com/questions/3221996/set-target-properties-called-with-incorrect-number-of-arguments/3222126


#]=]

set(Opticks_MODULE  "${CMAKE_CURRENT_LIST_FILE}")
include(GNUInstallDirs)

if(Opticks_VERBOSE)
    message(STATUS "${Opticks_MODULE} : Opticks_VERBOSE : ${Opticks_VERBOSE} ")

    foreach(_dir ${CMAKE_MODULE_PATH})
        message(STATUS "${Opticks_MODULE} : CMAKE_MODULE_PATH _dir : ${_dir} ")
    endforeach() 

    set(_prefix_list)
    string(REPLACE ":" ";" _prefix_list $ENV{CMAKE_PREFIX_PATH})
    foreach(_prefix ${_prefix_list})
        message(STATUS "${Opticks_MODULE} : CMAKE_PREFIX_PATH _prefix : ${_prefix} ")
    endforeach() 
endif()

find_package(G4OK CONFIG QUIET)

if(G4OK_FOUND)
    add_compile_definitions(WITH_G4OPTICKS)

    if(Opticks_VERBOSE)
        message(STATUS "${Opticks_MODULE} : PLog_INCLUDE_DIR:${PLog_INCLUDE_DIR} ")
    endif()

    include_directories(${PLog_INCLUDE_DIR})  ## WHY NOT AUTOMATIC ? Maybe because plog is header only ?

    set(Opticks_FOUND "YES") 
else()
    set(Opticks_FOUND "NO")
endif()


if(Opticks_VERBOSE)
    message(STATUS "${Opticks_MODULE} : Opticks_FOUND   : ${Opticks_FOUND} ")
endif()

