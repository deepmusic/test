
#include "logger.h"

logger global_logger;

logger::logger()
{
    QueryPerformanceFrequency(&freq_);
    memset(enabled_, 0, MAX_LOG_ITEMS*sizeof(bool));
    memset(acc_times_, 0, MAX_LOG_ITEMS*sizeof(bool));
    memset(counts_, 0, MAX_LOG_ITEMS*sizeof(bool));
}

void logger::set_log_item(unsigned int index, const char* name)
{
    if (index < MAX_LOG_ITEMS) {
        enabled_[index] = true;
        names_[index] = std::string(name);
    }
}

void logger::start_log(unsigned int index)
{
    if (index < MAX_LOG_ITEMS) {
        QueryPerformanceCounter(&tick_start_[index]);
    }
}

void logger::stop_log(unsigned int index)
{
    if (index < MAX_LOG_ITEMS) {
        LARGE_INTEGER tick;
        QueryPerformanceCounter(&tick);
        acc_times_[index] += 1000.0f * (tick.QuadPart - tick_start_[index].QuadPart) / freq_.QuadPart;
        counts_[index]++;
    }
}

void logger::print_log()
{
    for (int i = 0; i < MAX_LOG_ITEMS; i++) {
        if (enabled_[i]) {
            printf("[%d] %s:\t%d counts\t%.2f ms\n", i, names_[i].c_str(), counts_[i], acc_times_[i]);
        }
    }
}