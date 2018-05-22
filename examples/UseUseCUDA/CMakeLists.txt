cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseUseCUDA)
project(${name} VERSION 0.1.0)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(UseCUDA 0.1.0 REQUIRED)

include(EchoTarget)
echo_target_std(Opticks::UseCUDA)

add_executable( ${name} ${name}.cc )
target_link_libraries(${name} PUBLIC Opticks::UseCUDA )
install(TARGETS ${name} DESTINATION lib)

