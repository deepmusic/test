
#include "logger.h"

logger global_logger;

logger::logger()
{
  #ifdef _MSC_VER
    QueryPerformanceFrequency(&freq_);
    memset(enabled_, 0, MAX_LOG_ITEMS*sizeof(bool));
    memset(acc_times_, 0, MAX_LOG_ITEMS*sizeof(bool));
    memset(counts_, 0, MAX_LOG_ITEMS*sizeof(bool));
  #else
    return;
  #endif
}

void logger::set_log_item(unsigned int index, const char* name)
{
  #ifdef _MSC_VER
    if (index < MAX_LOG_ITEMS) {
        enabled_[index] = true;
        names_[index] = std::string(name);
    }
  #else
    return;
  #endif
}

void logger::start_log(unsigned int index)
{
  #ifdef _MSC_VER
    if (index < MAX_LOG_ITEMS) {
        QueryPerformanceCounter(&tick_start_[index]);
    }
  #else
    return;
  #endif
}

void logger::stop_log(unsigned int index)
{
  #ifdef _MSC_VER
    if (index < MAX_LOG_ITEMS) {
        LARGE_INTEGER tick;
        QueryPerformanceCounter(&tick);
        acc_times_[index] += 1000.0f * (tick.QuadPart - tick_start_[index].QuadPart) / freq_.QuadPart;
        counts_[index]++;
    }
  #else
    return;
  #endif
}

void logger::print_log()
{
  #ifdef _MSC_VER
    for (int i = 0; i < MAX_LOG_ITEMS; i++) {
        if (enabled_[i]) {
            printf("[%d] %s:\t%d counts\t%.2f ms\n", i, names_[i].c_str(), counts_[i], acc_times_[i]);
        }
    }
  #else
    return;
  #endif
}
