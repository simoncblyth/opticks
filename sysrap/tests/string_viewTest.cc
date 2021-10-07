/**
name=string_viewTest ; gcc $name.cc -std=c++17 -lstdc++ -o /tmp/$name && /tmp/$name

https://www.learncpp.com/cpp-tutorial/an-introduction-to-stdstring_view/

**/


#include <iostream>
#include <string_view>

int main()
{
  std::string_view text{ "hello" }; // view the text "hello", which is stored in the binary
  std::string_view str{ text }; // view of the same "hello"
  std::string_view more{ str }; // view of the same "hello"

  std::cout << text << ' ' << str << ' ' << more << '\n';

  return 0;
}
