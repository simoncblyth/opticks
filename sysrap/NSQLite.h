#pragma once
/**
NSQLite.h
===========

Convenience wrapper structs for using sqlite3.h
For usage examples see ~/np/tests/NSQLite_test.cc

Background

* https://sqlite.org/cintro.html

Find the header::

    rpm -ql sqlite-devel
    /usr/include/sqlite3.h
    /usr/include/sqlite3ext.h
    /usr/lib64/libsqlite3.so
    /usr/lib64/pkgconfig/sqlite3.pc


**/

#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <chrono>
#include <sqlite3.h>

struct OLD_NSQLiteStmt
{
    sqlite3_stmt* stmt = nullptr;
    OLD_NSQLiteStmt(sqlite3* db, const char* sql) {
        sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    }
    ~OLD_NSQLiteStmt() { sqlite3_finalize(stmt); }
};


struct NSQLiteStmt
{
    sqlite3_stmt* stmt = nullptr;
    sqlite3* db_ptr = nullptr;

    NSQLiteStmt(sqlite3* db, const char* sql) : db_ptr(db) {
        sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    }
    ~NSQLiteStmt() { sqlite3_finalize(stmt); }

    void bind_param(int index, int val) { sqlite3_bind_int(stmt, index, val); }
    void bind_param(int index, int64_t val) {       sqlite3_bind_int64(stmt, index, val); }
    void bind_param(int index, sqlite3_int64 val) { sqlite3_bind_int64(stmt, index, val); }
    void bind_param(int index, const char* val) { sqlite3_bind_text(stmt, index, val, -1, SQLITE_TRANSIENT); }
    void bind_param(int index, const std::string& val) { sqlite3_bind_text(stmt, index, val.c_str(), -1, SQLITE_TRANSIENT); }
    void bind_param(int index, double val) { sqlite3_bind_double(stmt, index, val); }
    void bind_param(int index, float val) {  sqlite3_bind_double(stmt, index, static_cast<double>(val));}
    // 3. NEW: Time/Date Support (Converts to Unix Epoch Integer)
    void bind_param(int index, const std::chrono::system_clock::time_point& val) {
        auto duration = val.time_since_epoch();
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
        sqlite3_bind_int64(stmt, index, static_cast<sqlite3_int64>(seconds));
    }

    template<typename... Args>
    bool execute(Args&&... args) {
        if (!stmt) return false;
        sqlite3_reset(stmt);

        int index = 1;
        (bind_param(index++, std::forward<Args>(args)), ...);
        // C++17 Fold Expression : specifically a "Unary Right Fold" using the comma operator
        // essentially compiler yields the needed sequence of bind_param calls for
        // each of the variadic arguments

        if (sqlite3_step(stmt) != SQLITE_DONE) {
            std::cerr << "Execution failed: " << sqlite3_errmsg(db_ptr) << std::endl;
            return false;
        }
        return true;
    }
};


/**
NSQLite
========

TODO: add exec of sql from file, eg for defining schema of collection of tables

**/

struct NSQLite
{
    char* err ;
    sqlite3* db;
    NSQLite(const char* filename);

    void exec(const char* sql );
    void exec_cb(const char* sql );
    void queryColumns(const char* sql);

    static int callback(void *data, int count, char **argv, char **columnNames);
    static std::string ReadStringDirect(const char* path);
};

inline int NSQLite::callback(void *data, int count, char **argv, char **columnNames) // static
{
   for(int i = 0; i<count; i++) printf("NSQLite::callback %s = %s\n", columnNames[i], argv[i]);
   printf("\n");
   return 0;
}

std::string NSQLite::ReadStringDirect(const char* path)  // static
{
    std::ifstream ifs(path, std::ios::in | std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) return {};

    std::ifstream::pos_type fileSize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::string str;
    str.resize(static_cast<size_t>(fileSize));

    ifs.read(str.data(), fileSize);

    return str;
}


/**
TODO: try using ":memory:" for filename when testing - for an in memory DB
**/


inline NSQLite::NSQLite(const char* filename)
    :
    err(nullptr),
    db(nullptr)
{
    sqlite3* database = nullptr ;
    int rc = sqlite3_open(filename, &database);
    if(rc != SQLITE_OK) {
        printf("NSQLite::NSQLite  could not be opened %s \n", sqlite3_errmsg(database));
    }
    else
    {
        db = database ;
    }
}

inline void NSQLite::exec(const char* sql )
{
    int rc = sqlite3_exec(db, sql, NULL, 0, &err);
    if(rc != SQLITE_OK) {
        printf("NSQLite::exec - error occured %s\n", err);
        sqlite3_free(err);
        err = nullptr ;
    } else {
        printf("NSQLite::exec successful \n");
    }
}

inline void NSQLite::exec_cb(const char* sql )
{
    int rc = sqlite3_exec(db, sql, callback, 0, &err);
    if(rc != SQLITE_OK) {
        printf("NSQLite::exec_cb - error occured %s\n", err);
        sqlite3_free(err);
        err = nullptr ;
    } else {
        printf("NSQLite::exec_cb successful \n");
    }
}

inline void NSQLite::queryColumns(const char* sql)
{
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if(rc != SQLITE_OK)
    {
        printf("NSQLite::queryColumns error occurred: %s", sqlite3_errmsg(db));
    }
    else
    {
        int num_col = sqlite3_column_count(stmt);
        printf("NSQLite::queryColumns num_col: %d\n\n", num_col);
        for(int i=0; i<num_col; i++)
        {
            const  char* col =  sqlite3_column_name(stmt, i);
            printf("col: %s\n", col);
        }
    }
    sqlite3_finalize(stmt);
}



