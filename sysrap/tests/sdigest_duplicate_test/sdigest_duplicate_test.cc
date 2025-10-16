#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include <thread>
#include <mutex>

#include "NP.hh"
#include "ssys.h"
#include "sdigest.h"




void process_chunk_0(
    const NP* hit,
    size_t start,
    size_t end,
    std::unordered_set<std::string>& seen,
    std::unordered_set<std::string>& duplicates,
    std::mutex& mtx)
{
    std::unordered_set<std::string> local_seen;
    std::unordered_set<std::string> local_duplicates;

    for (size_t i = start; i < end; ++i)
    {
        std::string digest = sdigest::Item(hit, i);
        if(local_seen.count(digest))
        {
            local_duplicates.insert(digest);
        }
        else
        {
            local_seen.insert(digest);
        }
    }

    // Merge with global sets (thread-safe)
    std::lock_guard<std::mutex> lock(mtx);
    for (const auto& digest : local_duplicates) {
        duplicates.insert(digest);
    }
    for (const auto& digest : local_seen) {
        if (seen.count(digest)) {
            duplicates.insert(digest);
        } else {
            seen.insert(digest);
        }
    }
}



void process_chunk_1(
    const NP* hit,
    size_t start,
    size_t end,
    std::unordered_set<std::string>& local_seen,
    std::unordered_set<std::string>& local_duplicates)
{
    for (size_t i = start; i < end; ++i)
    {
        std::string digest = sdigest::Item(hit, i);
        if (local_seen.count(digest)) {
            local_duplicates.insert(digest);
        } else {
            local_seen.insert(digest);
        }
    }
}











int main()
{
    // Load NumPy array
    const char* path = "$HITFOLD/hit.npy";

    std::cout
        << U::Log()
        << " [load " << path
        << "\n"
        ;

    NP* hit = NP::Load(path);

    std::cout
        << U::Log()
        << " ]load " << path
        << "\n"
        ;


    if (!hit) {
        std::cerr << "Failed to load " << path << std::endl;
        return 1;
    }

    size_t num_items = hit->shape[0]; // 598640516
    size_t num_threads = ssys::getenvint("NPROC", 10) ;
    size_t chunk_size = (num_items + num_threads - 1) / num_threads; // ~10.7M hits (~6.8 GB)
    size_t num_chunks = (num_items + chunk_size - 1) / chunk_size;

    std::cout
        << U::Log()
        << " hit " << hit->sstr()
        << " num_items " << num_items
        << " num_threads[NPROC] " << num_threads
        << " chunk_size " << chunk_size
        << " num_chunks " << num_chunks
        << "\n"
        ;


    /**
    std::unordered_set<std::string> seen_hashes;
    std::unordered_set<std::string> duplicate_hashes;
    std::mutex mtx;
    **/


    std::vector<std::unordered_set<std::string>> all_seen(num_chunks);
    std::vector<std::unordered_set<std::string>> all_duplicates(num_chunks);
    std::vector<std::thread> threads;

    for (size_t i = 0; i < num_chunks; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min(start + chunk_size, num_items);

        std::cout
            << U::Log()
            << " i " << std::setw(3) << i
            << " start " << std::setw(10) << start
            << " end " << std::setw(10) << end
            << "\n"
            ;

        /*
        threads.emplace_back(process_chunk_0, hit, start, end,
                            std::ref(seen_hashes), std::ref(duplicate_hashes), std::ref(mtx));
        */

        threads.emplace_back(process_chunk_1, hit, start, end,
                            std::ref(all_seen[i]), std::ref(all_duplicates[i]));
    }

    std::cout << U::Log(" before join \n") ;
    for (auto& t : threads)
    {
        std::cout << U::Log(" join \n") ;
        t.join();
    }


    std::cout << U::Log("merge \n") ;

    std::unordered_set<std::string> seen_hashes;
    std::unordered_set<std::string> duplicate_hashes;
    for (size_t i = 0; i < num_chunks; ++i)
    {
        for (const auto& digest : all_duplicates[i]) {
            duplicate_hashes.insert(digest);
        }
        for (const auto& digest : all_seen[i]) {
            if (seen_hashes.count(digest)) {
                duplicate_hashes.insert(digest);
            } else {
                seen_hashes.insert(digest);
            }
        }
    }

    std::cout
        << U::Log()
        << "Seen " << seen_hashes.size()
        << "Found " << duplicate_hashes.size()
        << " unique duplicated 4x4 hits."
        << std::endl
        ;

    delete hit;
    return 0;
}
