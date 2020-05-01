cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0063) #Honor visibility properties for all target types
    cmake_policy(SET CMP0063 NEW)
endif()


#[=[
UsePLogChained
================

Copy in the Chained sample from upstream latest:

    cd 
    git clone https://github.com/SergiusTheBest/plog.git 

    cd ~/opticks/examples
    cp -r ~/plog/samples/Chained/ChainedLib UsePLogChained/
    cp -r ~/plog/samples/Chained/ChainedApp UsePLogChained/
    cp ~/plog/samples/Chained/CMakeLists.txt UsePLogChained/

CMake setup adapted from::
 
   ~/plog/CMakeLists.txt
   ~/plog/samples/CMakeLists.txt

#]=]

project(UsePLogChained)
include(GNUInstallDirs)

if(UNIX AND NOT APPLE)
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
elseif(APPLE)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()


add_library(plog INTERFACE)
target_include_directories(plog
    INTERFACE
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
    $<BUILD_INTERFACE:${OPTICKS_PREFIX}/externals/plog/include>
)

add_library(plog::plog ALIAS plog)


if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_COMPILER_IS_CLANGXX 1)
endif ()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX")
    string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANGXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wconversion -Wnon-virtual-dtor -Wundef -pedantic -Werror")

    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden ")
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden ")

    # below is a higher level way of doing the above which is not specific to gcc 
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
endif()

#[=[
Commenting the visibility causes::

/tmp/blyth/opticks/lib/ChainedApp
ChainedApp: /home/blyth/local/opticks/externals/plog/include/plog/Logger.h:22: plog::Logger<instance>& plog::Logger<instance>::addAppender(plog::IAppender*) [with int instance = 0]: Assertion `appender != this' failed.
./go.sh: line 43: 41877 Aborted                 (core dumped) $bin
#]=]





add_executable(ChainedApp ChainedApp/Main.cpp)
target_link_libraries(ChainedApp ChainedLib plog)

#set_target_properties(ChainedApp PROPERTIES FOLDER Samples/Chained)
# set PLOG to local, so instances will not be shared across modules
#target_compile_definitions(ChainedApp PRIVATE PLOG_LOCAL)

add_library(ChainedLib SHARED ChainedLib/Main.cpp)
target_link_libraries(ChainedLib plog)

#set_target_properties(ChainedLib PROPERTIES FOLDER Samples/Chained)
# set PLOG to local, so instances will not be shared across modules
#target_compile_definitions(ChainedLib PRIVATE PLOG_LOCAL)


install(TARGETS ChainedLib DESTINATION lib)
install(TARGETS ChainedApp DESTINATION lib)


