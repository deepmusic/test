
#define MAX_LOG_ITEMS   100

#include <string>
#ifdef _MSC_VER
  #include <Windows.h>
#endif

class logger {

public:
    logger();
    ~logger() { }

    void set_log_item(unsigned int index, const char* name);
    void start_log(unsigned int index);
    void stop_log(unsigned int index);
    void print_log();

private:
    bool            enabled_[MAX_LOG_ITEMS];
    float           acc_times_[MAX_LOG_ITEMS];
    unsigned int    counts_[MAX_LOG_ITEMS];
    std::string     names_[MAX_LOG_ITEMS];
  #ifdef _MSC_VER
    LARGE_INTEGER freq_;
    LARGE_INTEGER tick_start_[MAX_LOG_ITEMS];
  #endif
};

extern logger global_logger;

enum {
    PVANET_DETECT,
    IMG2INPUT,
    FORWARD_NET_INCEPTION,
    FORWARD_NET_RCNN,
    FORWARD_CONV_LAYER,
    FORWARD_INPLACE_RELU_LAYER,
    FORWARD_POOL_LAYER,
    FORWARD_DECONV_LAYER,
    FORWARD_CONCAT_LAYER,
    FORWARD_PROPOSAL_LAYER,
    FORWARD_ROIPOOL_LAYER,
    FORWARD_FC_LAYER
};
