// This header file was auto-generated by ClassMate++
// Created: 22 Feb 2019 8:55:26 pm
// Copyright (c) 2019, HurleyWorks

#pragma once

// NO MORE TRACING  Yay
#define TRACE(x)

const LEVELS TESTING{INFO.value + 1, "TESTING"};
const LEVELS CRITICAL{WARNING.value + 1, "CRTICAL"};

#if defined(_WIN32) || defined(_WIN64)
#define __FUNCTION_NAME__ __func__
#define _FN_ __FUNCTION_NAME__
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#undef near
#undef far
#undef RGB
#endif


// makes it illegal to copy a derived class
// https://github.com/heisters/libnodes
struct Noncopyable
{
 protected:
    Noncopyable() = default;
    ~Noncopyable() = default;
    Noncopyable (const Noncopyable&) = delete;
    Noncopyable& operator= (const Noncopyable&) = delete;
};

// provides derived classes with automatically assigned,
// globally unique numeric identifiers
// https://github.com/heisters/libnodes
class HasId
{
 public:
    HasId() :
        mId (++sId)
    {
        //LOG (DBUG) << mId;
    }

    ItemID id() const { return mId; }
    void setID (ItemID itemID) { mId = itemID; }

    void staticReset (int id = 0) { sId = id; }

 protected:
    static ItemID sId;
    ItemID mId;
};

// from the Code Blacksmith
// https://www.youtube.com/watch?v=GV0JMHOpoEw
class ScopedStopWatch
{
 public:
    using Clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                                     std::chrono::high_resolution_clock,
                                     std::chrono::steady_clock>;
    ScopedStopWatch (const char function[] = "unknown function") :
        func (function)
    {
    }
    ~ScopedStopWatch()
    {
        LOG (DBUG) << "\n"
                   << func << " took " << std::chrono::duration_cast<std::chrono::milliseconds> (Clock::now() - start).count() << " milliseconds";
    }

 private:
    const char* func = nullptr;
    Clock::time_point start = Clock::now();
};

// store and retrieve any type from a map
template <class PROPERTY>
class AnyValue
{
    using ValueMap = std::unordered_map<int, std::any>;

 public:
    AnyValue() = default;
    ~AnyValue() = default;

    void addDefault (const PROPERTY& key, const std::any& value) { map_.insert (std::make_pair (key, value)); }
    void setValue (const PROPERTY& key, const std::any& value)
    {
        auto it = map_.find (key);
        if (it == map_.end())
            map_.insert (std::make_pair (key, value));
        else
            it->second = value;
    }

    template <typename T>
    T& getRef (const PROPERTY& key) { return std::any_cast<T> (getValue (key)); }

    template <typename T>
    T getVal (const PROPERTY& key) { return std::any_cast<T> (getValue (key)); }

    template <typename T>
    T* getPtr (const PROPERTY& key) { return std::any_cast<T> (&getValue (key)); }

 private:
    ValueMap map_;
    std::any empty_;

    std::any& getValue (const PROPERTY& key)
    {
        auto it = map_.find (key);
        if (it != map_.end())
            return it->second;

        return empty_;
    }

}; // end class AnyValue


static std::string readTxtFile (const std::filesystem::path& filepath)
{
    std::ifstream ifs;
    ifs.open (filepath, std::ios::in);
    if (ifs.fail())
        return "";

    std::stringstream sstream;
    sstream << ifs.rdbuf();

    return std::string (sstream.str());
}

static std::vector<char> readBinaryFile (const std::filesystem::path& filepath)
{
    std::vector<char> ret;

    std::ifstream ifs;
    ifs.open (filepath, std::ios::in | std::ios::binary | std::ios::ate);
    if (ifs.fail())
        return std::move (ret);

    std::streamsize fileSize = ifs.tellg();
    ifs.seekg (0, std::ios::beg);

    ret.resize (fileSize);
    ifs.read (ret.data(), fileSize);

    return std::move (ret);
}


