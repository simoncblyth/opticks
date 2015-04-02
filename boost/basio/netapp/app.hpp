#pragma once

#include <string>
#include <vector>


class App {
public:
   App();

   void on_msg(std::string msg);
   void on_npy(std::vector<int> shape, std::vector<float> data, std::string metadata);

};


