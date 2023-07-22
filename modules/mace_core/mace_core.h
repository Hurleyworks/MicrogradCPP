#pragma once

#include <unordered_map>
#include <unordered_set>
#include <array>
#include <queue>
#include <stack>
#include <fstream>
#include <set>
#include <vector>
#include <sstream>
#include <random>
#include <chrono>
#include <thread>
#include <ctime>
#include <string>
#include <iostream>
#include <stdexcept>
#include <assert.h>
#include <limits>
#include <algorithm>
#include <functional>
#include <stdint.h>
#include <any>
#include <filesystem>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <variant>
#include <future>
#include <semaphore>
#include <concepts>

// NO MORE TRACING  Yay
#define TRACE(x)

using ItemID = int64_t;

#ifdef __clang__
#include <experimental/coroutine>
#define COROUTINE_NAMESPACE std::experimental
#else
#include <coroutine>
#define COROUTINE_NAMESPACE std
#endif

// g3log
#include <g3log/g3log.hpp>
#include <g3log/logworker.hpp>

// random
#include <random/include/random.hpp>
using RandoM = effolkronium::random_static;

// json
#include <json/json.hpp>
using nlohmann::json;

// some useful tools and defines outside mace namespace
#include "excludeFromBuild/basics/Util.h"
#include "excludeFromBuild/ai/Micrograd.h"

namespace mace
{
	#include "excludeFromBuild/basics/StringUtil.h"

} // namespace mace
