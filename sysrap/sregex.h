#pragma once

#include <string>
#include <regex>
#include <cstdlib>
//#include <optional>

struct sregex
{
    std::regex re;
    bool valid = false;

    sregex(const char* env_var)
    {
        const char* pattern = std::getenv(env_var);
        if (pattern)
        {
            try {
                re = std::regex(pattern, std::regex::optimize);
                valid = true;
            } catch (const std::regex_error& e) {
                std::cerr << "Regex Error: " << e.what() << " in " << env_var << std::endl;
            }
        } else {
            std::cerr << "Environment variable " << env_var << " not set." << std::endl;
        }
    }

    bool matches(const std::string& input) const
    {
        if (!valid) return false;
        return std::regex_match(input, re);
    }
};


