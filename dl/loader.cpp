#include "layer.h"
#include <stdio.h>
#include <string.h>

#define MAX_NUM_TYPES 20

const static char* gs_types[MAX_NUM_TYPES] = {
  "__UNKNOWN_TYPE__",
  "INT_VAL",
  "REAL_VAL",
  "CONST_VAL",
  "NAME",
  "OPTION",
  "BLOCK",
  "{",
  "}",
  0,
};

const static int UNKNOWN_TYPE = 0;
const static int INT_VAL = 1;
const static int REAL_VAL = 2;
const static int CONST_VAL = 3;
const static int NAME = 4;
const static int OPTION = 5;
const static int BLOCK = 6;
const static int BLOCK_BEGIN = 7;
const static int BLOCK_END = 8;


#define MAX_NUM_WORDS 1024

typedef struct HashEntry_
{
  const char* const key;
  int type;
} HashEntry;

static HashEntry gs_reserved[MAX_NUM_WORDS] = {
  { "__UNDEFINED__", UNKNOWN_TYPE },
  //-----------------------------------
  { "param", BLOCK },
  { "std", OPTION },
  //-----------------------------------
  { "bias_filler", BLOCK },
  { "concat_param", BLOCK },
  { "convolution_param", BLOCK },
  { "dropout_param", BLOCK },
  { "inner_product_param", BLOCK },
  { "input", BLOCK },
  { "input_shape", BLOCK },
  { "layer", BLOCK },
  { "loss_param", BLOCK },
  { "pooling_param", BLOCK },
  { "proposal_param", BLOCK },
  { "reshape_param", BLOCK },
  { "roi_pooling_param", BLOCK },
  { "shape", BLOCK },
  { "weight_filler", BLOCK },
  //-----------------------------------
  { "{", BLOCK_BEGIN },
  { "}", BLOCK_END },
  //-----------------------------------
  { "axis", OPTION },
  { "base_size", OPTION },
  { "bias_term", OPTION },
  { "bottom", OPTION },
  { "class_name", OPTION },
  { "conf_thresh", OPTION },
  { "copys", OPTION },
  { "decay_mult", OPTION },
  { "dim", OPTION },
  { "dropout_ratio", OPTION },
  { "feat_stride", OPTION },
  { "group", OPTION },
  { "ignore_label", OPTION },
  { "kernel_h", OPTION },
  { "kernel_w", OPTION },
  { "kernel_size", OPTION },
  { "lr_mult", OPTION },
  { "max_size", OPTION },
  { "mean_value", OPTION },
  { "min_size", OPTION },
  { "multiple", OPTION },
  { "name", OPTION },
  { "nms_thresh", OPTION },
  { "normalize", OPTION },
  { "num_output", OPTION },
  { "pad", OPTION },
  { "pad_h", OPTION },
  { "pad_w", OPTION },
  { "pool", OPTION },
  { "pooled_h", OPTION },
  { "pooled_w", OPTION },
  { "post_nms_topn", OPTION },
  { "pre_nms_topn", OPTION },
  { "ratio", OPTION },
  { "scale", OPTION },
  { "scale_train", OPTION },
  { "spatial_scale", OPTION },
  { "stride", OPTION },
  { "stride_h", OPTION },
  { "stride_w", OPTION },
  { "top", OPTION },
  { "type", OPTION },
  { "value", OPTION },
  //-----------------------------------
  { "\"ReLU\"", NAME },
  //-----------------------------------
  { "\"Concat\"", NAME },
  { "\"Convolution\"", NAME },
  { "\"Deconvolution\"", NAME },
  { "\"Dropout\"", NAME },
  { "\"InnerProduct\"", NAME },
  { "\"Pooling\"", NAME },
  { "\"Proposal\"", NAME },
  { "\"Reshape\"", NAME },
  { "\"ROIPooling\"", NAME },
  { "\"Softmax\"", NAME },
  //-----------------------------------
  { "false", CONST_VAL },
  { "true", CONST_VAL },
  { "MAX", CONST_VAL },
  //-----------------------------------
  { 0, 0 },
};

static int gs_reserved_indices[MAX_NUM_WORDS] = { 0, };


static
unsigned int str2hash(const char* const str)
{
  const unsigned char* p_str = (unsigned char*)str;
  unsigned short hash = 5381;
  unsigned short ch;

  // for ch = 0, ..., strlen(str)-1
  while (ch = *(p_str++)) {
    // hash = hash * 33 + ch
    hash = ((hash << 5) + hash) + ch;
  }

  return hash;
}

static
int find_hash_table(const char* const key,
                    const HashEntry* const entries,
                    const int* const indices)
{
  unsigned int hash = str2hash(key) % MAX_NUM_WORDS;

  while (indices[hash]) {
    if (strcmp(key, entries[indices[hash]].key) == 0) {
      return indices[hash];
    }
    hash = (hash == MAX_NUM_WORDS - 1) ? 0 : hash + 1;
  }

  return 0;
}

static
int str2type(const char* const word)
{
  {
    int index = find_hash_table(word, gs_reserved, gs_reserved_indices);
    if (index) {
      return gs_reserved[index].type;
    }
  }

  {
    if (word[0] == '"' || word[0] == '\'') {
      const char* p_word = word;
      while (*(++p_word));
      if (*(p_word - 1) == '"' || *(p_word - 1) == '\'') {
        return CONST_VAL;
      }
    }
  }

  {
    const char* p_word = word;
    int ch;

    if (*p_word == '-') {
      ++p_word;
    }

    while (ch = *(p_word++)) {
      if (ch < '0' || ch > '9') {
        break;
      }
    }
    if (!ch) {
      return INT_VAL;
    }

    if (ch == '.') {
      while (ch = *(p_word++)) {
        if (ch < '0' || ch > '9') {
          break;
        }
      }
      if (!ch) {
        return REAL_VAL;
      }
    }
  }

  return UNKNOWN_TYPE;
}

static
void init_hash_table(const HashEntry* const entries,
                     int* const indices)
{
  int num_collisions = 0;

  for (int i = 0; i < MAX_NUM_WORDS; ++i) {
    if (!entries[i].key) {
      break;
    }

    {
      unsigned int hash = str2hash(entries[i].key) % MAX_NUM_WORDS;
      while (indices[hash]) {
        hash = (hash == MAX_NUM_WORDS - 1) ? 0 : hash + 1;
        ++num_collisions;
      }
      indices[hash] = i;
      printf("%s: %u, %d\n",
             entries[i].key, str2hash(entries[i].key) % MAX_NUM_WORDS, hash);
    }
  }

  printf("# collisions = %d\n", num_collisions);
}

static
void pop_spaces(FILE* fp)
{
  while (!feof(fp)) {
    char ch = (char)fgetc(fp);
    if (ch == '#') {
      while (!feof(fp) && fgetc(fp) != '\n');
    }
    else if (ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t' &&
             ch != ':') {
      ungetc(ch, fp);
      break;
    }
  }
}

static
int read_str(FILE* fp, char* const buf)
{
  char* p_buf = buf;
  int len = 0;

  while (!feof(fp)) {
    char ch = (char)fgetc(fp);

    if (ch == '{' || ch == '}') {
      if (len == 0) {
        p_buf[len++] = ch;
        break;
      }
      else {
        ungetc(ch, fp);
        break;
      }
    }

    if (ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t' ||
        ch == '#' || ch == ':') {
      ungetc(ch, fp);
      break;
    }

    p_buf[len++] = ch;
  }

  p_buf[len] = 0;
  return len;
}

void layer_begin(Net* net)
{
  net->layers[net->num_layers] = (Layer*)malloc(sizeof(Layer));
  printf("Layer %d creation started\n", net->num_layers);
}

void layer_end(Net* net)
{
  printf("Layer %d creation finished\n", net->num_layers);
  ++net->num_layers;
}

int main(int argc, char* argv[])
{
  FILE* fp = fopen(argv[1], "r");
  char buf[32];
  int args[50];
  char line[4096];
  int level = 0;
  int line_idx = 0;
  int word_idx = 0;

  init_hash_table(gs_reserved, gs_reserved_indices);

  while (!feof(fp)) {
    pop_spaces(fp);
    int len = read_str(fp, buf);
    if (len > 0) {
      int type = str2type(buf);
      switch (type) {
        case BLOCK:
        case OPTION:
          args[level] = find_hash_table(buf, gs_reserved, gs_reserved_indices);
          break;
        case BLOCK_BEGIN:
          if (gs_reserved[args[level]].type != BLOCK) {
            printf("[TYPE ERROR] %s(%s) is not BLOCK!\n", gs_reserved[args[level]].key, gs_types[gs_reserved[args[level]].type]);
          }
          else if (level == 0) {
            line_idx = 0;
            line_idx += sprintf(line + line_idx, "\n%s { ", gs_reserved[args[level]].key);
          }
          else {
            line_idx += sprintf(line + line_idx, "\n");
            for (int i = 0; i < level; ++i)
              line_idx += sprintf(line + line_idx, "  ");
            line_idx += sprintf(line + line_idx, "%s { ", gs_reserved[args[level]].key);
          }
          ++level;
          break;
        case BLOCK_END:
          --level;
          line_idx += sprintf(line + line_idx, "} ");
          if (level == 0) {
            printf("%s", line);
          }
          break;
        case NAME:
        case CONST_VAL:
          word_idx = find_hash_table(buf, gs_reserved, gs_reserved_indices);
          line_idx += sprintf(line + line_idx, "%s:%s(%s,%d) ", gs_reserved[args[level]].key, buf, gs_types[type], word_idx);
          break;
        case INT_VAL:
          line_idx += sprintf(line + line_idx, "%s:%d(%s) ", gs_reserved[args[level]].key, atoi(buf), gs_types[type]);
          break;
        case REAL_VAL:
          line_idx += sprintf(line + line_idx, "%s:%f(%s) ", gs_reserved[args[level]].key, (float)atof(buf), gs_types[type]);
          break;
        default:
          printf("Unknown command: %s\n", buf);
          break;
      }
    }
  }

  return 0;
}
