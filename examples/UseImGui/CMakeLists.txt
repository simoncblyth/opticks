cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseImGui)
project(${name} VERSION 0.1.0)

set(ImGui_VERBOSE ON)
find_package(ImGui REQUIRED MODULE)   

message(STATUS "ImGui_MODULE  : ${ImGui_MODULE} ")

include(EchoTarget)
#echo_pfx_vars(ImGui "FOUND;VERSION;CONFIG;INCLUDE;LIBRARY;DLL;INSTALL_DIR")
echo_target_std(Opticks::ImGui)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::ImGui )
install(TARGETS ${name}  DESTINATION lib)

