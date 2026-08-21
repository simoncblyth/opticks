#pragma once
/**
scode.h : Code loader that flattens included files into a string
=================================================================

Used from sysrap/SGLFW_Program.h to enable simple "#include" functionality
for glsl shader source files.

**/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <filesystem>

namespace fs = std::filesystem;

struct scode
{
    static std::string load(const fs::path& fold, const fs::path& name);
    static std::string load(const fs::path& filepath);

private:
    static std::string loadRecursive(const fs::path& filepath, std::unordered_set<std::string>& included_files);
    static fs::path resolvePath(const fs::path& base_dir, const std::string& include_filename);
};

inline std::string scode::load(const fs::path& fold, const fs::path& name)
{
    return load( fold / name );
}
inline std::string scode::load(const fs::path& filepath)
{
    std::unordered_set<std::string> included_files;
    fs::path target_path = fs::absolute(filepath);

    if (!fs::exists(target_path)) {
        std::cerr << "scode::Load - Failed to find top file: " << target_path << std::endl;
        return "";
    }

    return loadRecursive(target_path, included_files);
}


/**
scode::resolvePath
-------------------

Helper that checks current directory (base_dir), then parent directory (base_dir / "..")

NB uses fs::weakly_canonical as manu candidate include files will not exist and fs::weakly_canonical
can cope with that unlike fs::canonical

**/

inline fs::path scode::resolvePath(const fs::path& base_dir, const std::string& include_filename)
{
    // Search Candidate 1: Relative to the file currently being processed
    fs::path p1 = fs::weakly_canonical(base_dir / include_filename);
    if (fs::exists(p1)) {
        return p1;
    }

    // Search Candidate 2: One directory up relative to the current file
    fs::path p2 = fs::weakly_canonical(base_dir / ".." / include_filename);
    if (fs::exists(p2)) {
        return p2;
    }

    return {}; // Return empty path if not found in either location
}

inline std::string scode::loadRecursive(const fs::path& filepath, std::unordered_set<std::string>& included_files)
{
    // Canonicalize path to ensure robust cycle detection across relative path variants
    std::string canonical_key = fs::canonical(filepath).string();

    // Prevent infinite recursion loops (circular includes)
    if (included_files.count(canonical_key)) {
        return "";
    }
    included_files.insert(canonical_key);

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "scode::loadRecursive - Failed to open: " << filepath << std::endl;
        return "";
    }

    std::stringstream result;
    std::string line;
    fs::path base_dir = filepath.parent_path();

    while (std::getline(file, line)) {
        // Simple check for #include "..."
        if (line.rfind("#include", 0) == 0) { // Line starts with #include
            size_t first_quote = line.find('"');
            size_t last_quote = line.rfind('"');

            if (first_quote != std::string::npos && last_quote > first_quote) {
                std::string include_filename = line.substr(first_quote + 1, last_quote - first_quote - 1);

                // Resolve include path in target locations
                fs::path resolved_path = resolvePath(base_dir, include_filename);

                if (resolved_path.empty()) {
                    std::cerr << "scode::loadRecursive - Could not resolve include '" << include_filename
                              << "' referenced in " << filepath << std::endl;
                } else {
                    result << "// --- BEGIN INCLUDE: " << include_filename << " ---\n";
                    result << loadRecursive(resolved_path, included_files);
                    result << "// --- END INCLUDE: " << include_filename << " ---\n";
                    continue;
                }
            }
        }
        result << line << "\n";
    }
    return result.str();
}


