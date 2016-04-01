#include <stdio.h>
#include <string.h>

#define MAX_NUM_WORDS 4096

const static char* gs_words[MAX_NUM_WORDS] = {
  "__UNDEFINED__",
  "axis",
  "base_size",
  "bias_filler",
  "bias_term",
  "bottom",
  "class_name",
  "concat_param",
  "conf_thresh",
  "convolution_param",
  "decay_mult",
  "dim",
  "dropout_param",
  "dropout_ratio",
  "false",
  "feat_stride",
  "group",
  "ignore_label",
  "inner_product_param",
  "input",
  "input_shape",
  "layer",
  "loss_param",
  "kernel_h",
  "kernel_w",
  "kernel_size",
  "lr_mult",
  "max_size",
  "mean_value",
  "min_size",
  "multiple",
  "name",
  "nms_thresh",
  "normalize",
  "num_output",
  "pad",
  "pad_h",
  "pad_w",
  "param",
  "pool",
  "pooled_h",
  "pooled_w",
  "pooling_param",
  "post_nms_topn",
  "pre_nms_topn",
  "proposal_param",
  "ratio",
  "reshape_param",
  "roi_pooling_param",
  "scale",
  "scale_train",
  "shape",
  "spatial_scale",
  "std",
  "stride",
  "stride_h",
  "stride_w",
  "top",
  "true",
  "type",
  "value",
  "weight_filler",
  "MAX",
  0,
};

int gs_indices[MAX_NUM_WORDS] = { 0, };

#define MAX_NUM_TYPES 10

const static char* gs_types[MAX_NUM_TYPES] = {
  "__UNKNOWN_TYPE__",
  "INT_VAL",
  "REAL_VAL",
  "STRING",
  "RESERVED_WORD",
  "{",
  "}",
  0,
};

const static int gs_enum_unknown = 0;
const static int gs_enum_int = 1;
const static int gs_enum_real = 2;
const static int gs_enum_string = 3;
const static int gs_enum_word = 4;
const static int gs_enum_block_begin = 5;
const static int gs_enum_block_end = 6;

static
unsigned int str2hash(const char* const str)
{
  const unsigned char* p_str = (unsigned char*)str;
  unsigned int hash = 5381;
  unsigned int ch;

  // for ch = 0, ..., strlen(str)-1
  while (ch = *(p_str++)) {
    // hash = hash * 33 + ch
    hash = ((hash << 5) + hash) + ch;
  }

  return hash;
}

static
int find_word(const char* const str)
{
  unsigned int hash = str2hash(str) % MAX_NUM_WORDS;

  while (gs_indices[hash]) {
    if (strcmp(str, gs_words[gs_indices[hash]]) == 0) {
      return gs_indices[hash];
    }
    hash = (hash == MAX_NUM_WORDS - 1) ? 0 : hash + 1;
  }

  return 0;
}

static
int str2type(const char* const str)
{
  if (str[0] == '"' || str[0] == '\'') {
    const char* p_str = str;
    while (*(++p_str));
    if (*(p_str - 1) == '"' || *(p_str - 1) == '\'') {
      return gs_enum_string;
    }
  }

  if (str[0] == '{') {
    return gs_enum_block_begin;
  }

  if (str[0] == '}') {
    return gs_enum_block_end;
  }

  if (find_word(str) > 0) {
    return gs_enum_word;
  }

  {
    const char* p_str = str;
    int ch;

    if (*p_str == '-') {
      ++p_str;
    }

    while (ch = *(p_str++)) {
      if (ch < '0' || ch > '9') {
        break;
      }
    }
    if (!ch) {
      return gs_enum_int;
    }

    if (ch == '.') {
      while (ch = *(p_str++)) {
        if (ch < '0' || ch > '9') {
          break;
        }
      }
      if (!ch) {
        return gs_enum_real;
      }
    }
  }

  return gs_enum_unknown;
}

static
void init_parser(void)
{
  int num_collisions = 0;

  for (int i = 0; i < MAX_NUM_WORDS; ++i) {
    if (!gs_words[i]) {
      break;
    }

    {
      unsigned int hash = str2hash(gs_words[i]) % MAX_NUM_WORDS;
      while (gs_indices[hash]) {
        hash = (hash == MAX_NUM_WORDS - 1) ? 0 : hash + 1;
        ++num_collisions;
      }
      gs_indices[hash] = i;
      printf("%s: %u, %d\n",
             gs_words[i], str2hash(gs_words[i]) % MAX_NUM_WORDS, hash);
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

int main(int argc, char* argv[])
{
  FILE* fp = fopen(argv[1], "r");
  char buf[1024];

  init_parser();
  while (!feof(fp)) {
    pop_spaces(fp);
    int len = read_str(fp, buf);
    if (len > 0) {
      int word_idx = find_word(buf);
      printf("[%s] %s, %d, %s\n", gs_types[str2type(buf)], buf, word_idx, gs_words[word_idx]);
    }
  }

  return 0;
}
