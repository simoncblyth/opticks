


-- Build files have been written to: /data/simon/local/opticks/externals/imgui/imgui.build
Scanning dependencies of target ImGui
[ 20%] Building CXX object CMakeFiles/ImGui.dir/imgui.cpp.o
[ 40%] Building CXX object CMakeFiles/ImGui.dir/imgui_draw.cpp.o
/data/simon/local/opticks/externals/imgui/imgui/imgui_draw.cpp: In member function ‘void ImDrawList::ClearFreeMemory()’:
/data/simon/local/opticks/externals/imgui/imgui/imgui_draw.cpp:139:66: warning: ‘void* memset(void*, int, size_t)’ clearing an object of non-trivial type ‘ImVector<ImDrawChannel>::value_type’ {aka ‘struct ImDrawChannel’}; use assignment or value-initialization instead [-Wclass-memaccess]
         if (i == 0) memset(&_Channels[0], 0, sizeof(_Channels[0]));  // channel 0 is a copy of CmdBuffer/IdxBuffer, don't destruct again
                                                                  ^
In file included from /data/simon/local/opticks/externals/imgui/imgui/imgui_draw.cpp:15:
/data/simon/local/opticks/externals/imgui/imgui/imgui.h:1099:8: note: ‘ImVector<ImDrawChannel>::value_type’ {aka ‘struct ImDrawChannel’} declared here
 struct ImDrawChannel
        ^~~~~~~~~~~~~
/data/simon/local/opticks/externals/imgui/imgui/imgui_draw.cpp: In member function ‘void ImDrawList::ChannelsSplit(int)’:
/data/simon/local/opticks/externals/imgui/imgui/imgui_draw.cpp:264:51: warning: ‘void* memset(void*, int, size_t)’ clearing an object of non-trivial type ‘ImVector<ImDrawChannel>::value_type’ {aka ‘struct ImDrawChannel’}; use assignment or value-initialization instead [-Wclass-memaccess]
     memset(&_Channels[0], 0, sizeof(ImDrawChannel));
                                                   ^
In file included from /data/simon/local/opticks/externals/imgui/imgui/imgui_draw.cpp:15:
/data/simon/local/opticks/externals/imgui/imgui/imgui.h:1099:8: note: ‘ImVector<ImDrawChannel>::value_type’ {aka ‘struct ImDrawChannel’} declared here
 struct ImDrawChannel
        ^~~~~~~~~~~~~
/data/simon/local/opticks/externals/imgui/imgui/imgui_draw.cpp: In member function ‘void ImDrawList::ChannelsSetCurrent(int)’:
/data/simon/local/opticks/externals/imgui/imgui/imgui_draw.cpp:324:86: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of non-trivially copyable type ‘class ImVector<ImDrawCmd>’; use copy-assignment or copy-initialization instead [-Wclass-memaccess]
     memcpy(&_Channels.Data[_ChannelsCurrent].CmdBuffer, &CmdBuffer, sizeof(CmdBuffer)); // copy 12 bytes, four times
                                                                                      ^
In file included from /data/simon/local/opticks/externals/imgui/imgui/imgui_draw.cpp:15:
/data/simon/local/opticks/externals/imgui/imgui/imgui.h:802:7: note: ‘class ImVector<ImDrawCmd>’ declared here
 class ImVector
       ^~~~~~~~
/data/simon/local/opticks/externals/imgui/imgui/imgui_draw.cpp:325:86: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of non-trivially copyable type ‘class ImVector<short unsigned int>’; use copy-assignment or copy-initialization instead [-Wclass-memaccess]
     memcpy(&_Channels.Data[_ChannelsCurrent].IdxBuffer, &IdxBuffer, sizeof(IdxBuffer));
                                                                                      ^
In file included from /data/simon/local/opticks/externals/imgui/imgui/imgui_draw.cpp:15:
/data/simon/local/opticks/externals/imgui/imgui/imgui.h:802:7: note: ‘class ImVector<short unsigned int>’ declared here
 class ImVector
       ^~~~~~~~
/data/simon/local/opticks/externals/imgui/imgui/imgui_draw.cpp:327:86: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of non-trivially copyable type ‘class ImVector<ImDrawCmd>’; use copy-assignment or copy-initialization instead [-Wclass-memaccess]
     memcpy(&CmdBuffer, &_Channels.Data[_ChannelsCurrent].CmdBuffer, sizeof(CmdBuffer));
                                                                                      ^
In file included from /data/simon/local/opticks/externals/imgui/imgui/imgui_draw.cpp:15:
/data/simon/local/opticks/externals/imgui/imgui/imgui.h:802:7: note: ‘class ImVector<ImDrawCmd>’ declared here
 class ImVector
       ^~~~~~~~
/data/simon/local/opticks/externals/imgui/imgui/imgui_draw.cpp:328:86: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of non-trivially copyable type ‘class ImVector<short unsigned int>’; use copy-assignment or copy-initialization instead [-Wclass-memaccess]
     memcpy(&IdxBuffer, &_Channels.Data[_ChannelsCurrent].IdxBuffer, sizeof(IdxBuffer));
                                                                                      ^
In file included from /data/simon/local/opticks/externals/imgui/imgui/imgui_draw.cpp:15:
/data/simon/local/opticks/externals/imgui/imgui/imgui.h:802:7: note: ‘class ImVector<short unsigned int>’ declared here
 class ImVector
       ^~~~~~~~
/data/simon/local/opticks/externals/imgui/imgui/imgui.h: In instantiation of ‘void ImVector<T>::reserve(int) [with T = ImDrawChannel]’:
/data/simon/local/opticks/externals/imgui/imgui/imgui.h:836:92:   required from ‘void ImVector<T>::resize(int) [with T = ImDrawChannel]’
/data/simon/local/opticks/externals/imgui/imgui/imgui_draw.cpp:258:40:   required from here
/data/simon/local/opticks/externals/imgui/imgui/imgui.h:841:15: warning: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of non-trivially copyable type ‘struct ImDrawChannel’; use copy-assignment or copy-initialization instead [-Wclass-memaccess]
         memcpy(new_data, Data, (size_t)Size * sizeof(value_type));
         ~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/data/simon/local/opticks/externals/imgui/imgui/imgui.h:1099:8: note: ‘struct ImDrawChannel’ declared here
 struct ImDrawChannel
        ^~~~~~~~~~~~~
[ 60%] Building CXX object CMakeFiles/ImGui.dir/imgui_demo.cpp.o
/data/simon/local/opticks/externals/imgui/imgui/imgui_demo.cpp: In function ‘void ImGui::ShowTestWindow(bool*)’:
/data/simon/local/opticks/externals/imgui/imgui/imgui_demo.cpp:1349:13: warning: this ‘if’ clause does not guard... [-Wmisleading-indentation]
             if (ImGui::CollapsingHeader("Category A")) ImGui::Text("Blah blah blah"); ImGui::NextColumn();
             ^~
/data/simon/local/opticks/externals/imgui/imgui/imgui_demo.cpp:1349:87: note: ...this statement, but the latter is misleadingly indented as if it were guarded by the ‘if’
             if (ImGui::CollapsingHeader("Category A")) ImGui::Text("Blah blah blah"); ImGui::NextColumn();
                                                                                       ^~~~~
/data/simon/local/opticks/externals/imgui/imgui/imgui_demo.cpp:1350:13: warning: this ‘if’ clause does not guard... [-Wmisleading-indentation]
             if (ImGui::CollapsingHeader("Category B")) ImGui::Text("Blah blah blah"); ImGui::NextColumn();
             ^~
/data/simon/local/opticks/externals/imgui/imgui/imgui_demo.cpp:1350:87: note: ...this statement, but the latter is misleadingly indented as if it were guarded by the ‘if’
             if (ImGui::CollapsingHeader("Category B")) ImGui::Text("Blah blah blah"); ImGui::NextColumn();
                                                                                       ^~~~~
/data/simon/local/opticks/externals/imgui/imgui/imgui_demo.cpp:1351:13: warning: this ‘if’ clause does not guard... [-Wmisleading-indentation]
             if (ImGui::CollapsingHeader("Category C")) ImGui::Text("Blah blah blah"); ImGui::NextColumn();
             ^~
/data/simon/local/opticks/externals/imgui/imgui/imgui_demo.cpp:1351:87: note: ...this statement, but the latter is misleadingly indented as if it were guarded by the ‘if’
             if (ImGui::CollapsingHeader("Category C")) ImGui::Text("Blah blah blah"); ImGui::NextColumn();
                                                                                       ^~~~~
/data/simon/local/opticks/externals/imgui/imgui/imgui_demo.cpp: In member function ‘void ExampleAppConsole::Draw(const char*, bool*)’:
/data/simon/local/opticks/externals/imgui/imgui/imgui_demo.cpp:1944:9: warning: this ‘if’ clause does not guard... [-Wmisleading-indentation]
         if (ImGui::SmallButton("Add Dummy Error")) AddLog("[error] something went wrong"); ImGui::SameLine();
         ^~
/data/simon/local/opticks/externals/imgui/imgui/imgui_demo.cpp:1944:92: note: ...this statement, but the latter is misleadingly indented as if it were guarded by the ‘if’
         if (ImGui::SmallButton("Add Dummy Error")) AddLog("[error] something went wrong"); ImGui::SameLine();
                                                                                            ^~~~~
/data/simon/local/opticks/externals/imgui/imgui/imgui_demo.cpp:1989:13: warning: this ‘while’ clause does not guard... [-Wmisleading-indentation]
             while (input_end > InputBuf && input_end[-1] == ' ') input_end--; *input_end = 0;
             ^~~~~
/data/simon/local/opticks/externals/imgui/imgui/imgui_demo.cpp:1989:79: note: ...this statement, but the latter is misleadingly indented as if it were guarded by the ‘while’
             while (input_end > InputBuf && input_end[-1] == ' ') input_end--; *input_end = 0;
                                                                               ^
[ 80%] Building CXX object CMakeFiles/ImGui.dir/examples/opengl3_example/imgui_impl_glfw_gl3.cpp.o
[100%] Linking CXX shared library libImGui.so
[100%] Built target ImGui
Install the project...
-- Install configuration: "Debug"
-- Installing: /data/simon/local/opticks/externals/lib/libImGui.so
-- Set runtime path of "/data/simon/local/opticks/externals/lib/libImGui.so" to "$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/OptiX/lib64"
-- Installing: /data/simon/local/opticks/externals/include/ImGui/imgui.h
-- Installing: /data/simon/local/opticks/externals/include/ImGui/imconfig.h
-- Installing: /data/simon/local/opticks/externals/include/ImGui/imgui_impl_glfw_gl3.h
=== imgui-pc : path /data/simon/local/opticks/externals/lib/pkgconfig/ImGui.pc

