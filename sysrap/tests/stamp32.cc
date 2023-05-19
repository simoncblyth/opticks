// name=stamp32 ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

/**
Datetime corresponding to UINT32_MAX: Thu Jan  1 02:11:34 1970


Persisting stamps with 32 bits is not practical. 
**/

#include <iostream>
#include <chrono>
#include <ctime>

int main() {
    // Calculate the datetime corresponding to UINT32_MAX
    std::chrono::system_clock::time_point epochTime;
    std::chrono::system_clock::duration durationSinceEpoch = std::chrono::microseconds(UINT32_MAX);
    std::chrono::system_clock::time_point maxTime = epochTime + durationSinceEpoch;

    // Convert the time point to a C-style time
    std::time_t maxTimeT = std::chrono::system_clock::to_time_t(maxTime);

    // Convert to a string representation
    std::string maxTimeStr = std::ctime(&maxTimeT);

    std::cout << "Datetime corresponding to UINT32_MAX: " << maxTimeStr;

    return 0;
}

