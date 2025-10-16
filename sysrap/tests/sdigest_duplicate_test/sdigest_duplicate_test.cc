#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include <thread>
#include <mutex>
#include <array>
#include <functional> // for std::hash

#include "NP.hh"
#include "ssys.h"
#include "sdigest.h"

using Digest = std::array<unsigned char, 16>;



struct DigestHash 
{
    std::size_t operator()(const std::array<unsigned char, 16>& digest) const noexcept 
    {
        std::size_t seed = 0;
        // Treat the 16 bytes as four 32-bit chunks for better distribution
        for (size_t i = 0; i < 16; i += 4) 
        {
            // Combine four bytes into a 32-bit integer (little-endian)
            uint32_t chunk = (static_cast<uint32_t>(digest[i]) << 24) |
                             (static_cast<uint32_t>(digest[i + 1]) << 16) |
                             (static_cast<uint32_t>(digest[i + 2]) << 8) |
                             static_cast<uint32_t>(digest[i + 3]);
            // Combine into seed using a simple but effective mixing function
            seed ^= std::hash<uint32_t>{}(chunk) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};


using DigestSet = std::unordered_set<Digest,DigestHash> ;



void process_chunk_0(
    const NP* hit,
    size_t start,
    size_t end,
    DigestSet& seen,
    DigestSet& duplicates,
    std::mutex& mtx)
{
    DigestSet local_seen;
    DigestSet local_duplicates;

    for (size_t i = start; i < end; ++i)
    {
        Digest digest = sdigest::ItemRaw(hit, i);
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
    DigestSet& local_seen,
    DigestSet& local_duplicates)
{
    for (size_t i = start; i < end; ++i)
    {
        Digest digest = sdigest::ItemRaw(hit, i);
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




    std::vector<DigestSet> all_seen(num_chunks);
    std::vector<DigestSet> all_duplicates(num_chunks);
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

    DigestSet seen_hashes;
    DigestSet duplicate_hashes;

    for (size_t i = 0; i < num_chunks; ++i)
    {
        for (const auto& digest : all_duplicates[i]) 
        {
            duplicate_hashes.insert(digest);
        }
        for (const auto& digest : all_seen[i]) 
        {
            if (seen_hashes.count(digest)) 
            {
                duplicate_hashes.insert(digest);
            } 
            else 
            {
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
