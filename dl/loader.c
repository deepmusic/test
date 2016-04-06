#include "layer.h"
#include <stdio.h>
#include <string.h>

#define MAX_NUM_TYPES 20

const static char* gs_types[MAX_NUM_TYPES] = {
  "__UNKNOWN_TYPE__",
  "INT_VAL",
  "REAL_VAL",
  "STRING_VAL",
  "TENSOR_VAL",
  "ENTRY_VAL",
  "LAYER",
  "OPERATOR",
  0,
};

#define UNKNOWN_TYPE 0
#define INT_VAL 1
#define REAL_VAL 2
#define STRING_VAL 3
#define TENSOR_VAL 4
#define ENTRY_VAL 5
#define LAYER 6
#define OPERATOR 7


#define MAX_NUM_ENTRIES 65536

typedef struct HashEntry_
{
  char* p_name;
  char* p_key;
  int type;
  int num_values;
  void** p_values;
} HashEntry;


static HashEntry gs_all_entries[MAX_NUM_ENTRIES] = {
  { 0, 0, 0, 0, 0 },
};


unsigned int entry2hash(const char* const p_key,
                        const int type)
{
  const unsigned char* p_key_ = (unsigned char*)p_key;
  unsigned short hash = 5381;
  unsigned short ch;

  hash = ((hash << 5) + hash) + (unsigned short)type;
  // for ch = 0, ..., strlen(p_key_)-1
  while (ch = *(p_key_++)) {
    // hash = hash * 33 + ch
    hash = ((hash << 5) + hash) + ch;
  }

  return hash;
}

int is_two_to_the_n(const unsigned int val)
{
  unsigned int i = 1;
  while (i < val) {
    i *= 2;
  }
  return (i == val);
}

HashEntry* find_hash_entry(HashEntry* const entries,
                           const char* const p_key,
                           const int type)
{
  unsigned int hash = entry2hash(p_key, type) % MAX_NUM_ENTRIES;

  while (entries[hash].p_key) {
    if (strcmp(p_key, entries[hash].p_key) == 0) {
      return &entries[hash];
    }
    hash = (hash == MAX_NUM_ENTRIES - 1) ? 0 : hash + 1;
  }

  return NULL;
}

HashEntry* find_or_make_hash_entry(HashEntry* const entries,
                                   const char* const p_name,
                                   const char* const p_key,
                                   const int type)
{
  unsigned int hash = entry2hash(p_key, type) % MAX_NUM_ENTRIES;

  while (entries[hash].p_key) {
    if (strcmp(p_key, entries[hash].p_key) == 0) {
      return &entries[hash];
    }
    hash = (hash == MAX_NUM_ENTRIES - 1) ? 0 : hash + 1;
  }

  entries[hash].p_name = (char*)malloc((strlen(p_name) + 1) * sizeof(char));
  strcpy(entries[hash].p_name, p_name);
  entries[hash].p_key = (char*)malloc((strlen(p_key) + 1) * sizeof(char));
  strcpy(entries[hash].p_key, p_key);
  entries[hash].type = type;
  entries[hash].p_values = NULL;
  entries[hash].num_values = 0;

  return &entries[hash];
}

void append_value_to_hash_entry(HashEntry* const p_entry,
                                void* const p_value)
{
  if (!p_entry->p_values) {
    p_entry->p_values = (void**)malloc(sizeof(void*));
    p_entry->p_values[0] = p_value;
    p_entry->num_values = 1;
  }

  else if (is_two_to_the_n(p_entry->num_values)) {
    void** p_values_new =
        (void**)calloc(p_entry->num_values * 2, sizeof(void*));
    memcpy(p_values_new, p_entry->p_values,
           p_entry->num_values * sizeof(void*));
    free(p_entry->p_values);
    p_entry->p_values = p_values_new;
    p_entry->p_values[p_entry->num_values++] = p_value;
  }

  else {
    p_entry->p_values[p_entry->num_values++] = p_value;
  }
}

void print_hash_entry(const HashEntry* const p_entry, const int level)
{
  for (int i = 0; i < level; ++i) {
    printf("  ");
  }
  printf("Entry %s (%s, %s): ",
         p_entry->p_name, p_entry->p_key, gs_types[p_entry->type]);
  switch (p_entry->type) {
    case INT_VAL:
      printf("[");
      for (int i = 0; i < p_entry->num_values - 1; ++i) {
        printf("%d, ", *(((int**)p_entry->p_values)[i]));
      }
      printf("%d]\n",
          *(((int**)p_entry->p_values)[p_entry->num_values - 1]));
      break;
    case REAL_VAL:
      printf("[");
      for (int i = 0; i < p_entry->num_values - 1; ++i) {
        printf("%f, ", *(((real**)p_entry->p_values)[i]));
      }
      printf("%f]\n",
          *(((real**)p_entry->p_values)[p_entry->num_values - 1]));
      break;
    case STRING_VAL:
      printf("[");
      for (int i = 0; i < p_entry->num_values - 1; ++i) {
        printf("%s, ", ((const char**)p_entry->p_values)[i]);
      }
      printf("%s]\n",
          ((const char**)p_entry->p_values)[p_entry->num_values - 1]);
      break;
    case TENSOR_VAL:
      printf("[");
      for (int i = 0; i < p_entry->num_values - 1; ++i) {
        printf("%s, ", ((const Tensor**)p_entry->p_values)[i]->name);
      }
      printf("%s]\n",
        ((const Tensor**)p_entry->p_values)[p_entry->num_values - 1]->name);
      break;
    case ENTRY_VAL:
      if (p_entry->num_values == 0) {
        printf("0 value");
      }
      printf("\n");
      for (int i = 0; i < p_entry->num_values; ++i) {
        print_hash_entry(((const HashEntry**)p_entry->p_values)[i],
                         level + 1);
      }
      break;
    case LAYER:
      printf("[");
      for (int i = 0; i < p_entry->num_values - 1; ++i) {
        printf("%s, ", ((const Layer**)p_entry->p_values)[i]->name);
      }
      printf("%s]\n",
        ((const Layer**)p_entry->p_values)[p_entry->num_values - 1]->name);
      break;
    case OPERATOR:
      printf("[");
      for (int i = 0; i < p_entry->num_values - 1; ++i) {
        printf("%s, ", ((const char**)p_entry->p_values)[i]);
      }
      printf("%s]\n",
          ((const char**)p_entry->p_values)[p_entry->num_values - 1]);
      break;
    default:
      printf("%d values of unknown types\n", p_entry->num_values);
      break;
  }
}

int word2type(const char* const p_word)
{
  {
    if (p_word[0] == '"') {
      const char* p_word_ = p_word;
      while (*(++p_word_));
      if (*(p_word_ - 1) == '"') {
        return STRING_VAL;
      }
    }
  }

  {
    const char* p_word_ = p_word;
    int ch;

    if (*p_word_ == '-') {
      ++p_word_;
    }

    while (ch = *(p_word_++)) {
      if (ch < '0' || ch > '9') {
        break;
      }
    }
    if (!ch) {
      return INT_VAL;
    }

    if (ch == '.') {
      while (ch = *(p_word_++)) {
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

void pop_spaces(FILE* fp)
{
  while (!feof(fp)) {
    char ch = (char)fgetc(fp);
    if (ch == '#') {
      while (!feof(fp) && fgetc(fp) != '\n');
    }
    else if (ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t') {
      ungetc(ch, fp);
      break;
    }
  }
}

int read_word(FILE* fp, char* const p_buf)
{
  int len = 0;

  while (!feof(fp)) {
    char ch = (char)fgetc(fp);

    if (ch == '{' || ch == '}' || ch == ':') {
      if (len == 0) {
        p_buf[len++] = ch;
        break;
      }
      else {
        ungetc(ch, fp);
        break;
      }
    }

    else if (ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t' ||
             ch == '#') {
      ungetc(ch, fp);
      break;
    }

    else if (ch == '\'') {
      ch = '"';
    }

    p_buf[len++] = ch;
  }

  p_buf[len] = 0;
  return len;
}

void free_hash_entry(HashEntry* const p_entry)
{
  if (p_entry->p_key) {
    //print_hash_entry(p_entry, 0);
    free(p_entry->p_key);
  }
  if (p_entry->p_name) {
    free(p_entry->p_name);
  }
  if (p_entry->p_values) {
    if (p_entry->type != ENTRY_VAL) {
      for (int i = 0; i < p_entry->num_values; ++i) {
        if (p_entry->p_values[i]) {
          //printf("  free value %d\n", i);
          free(p_entry->p_values[i]);
        }
      }
    }
    free(p_entry->p_values);
  }
  memset(p_entry, 0, sizeof(HashEntry));
}

void free_hash_table(HashEntry* const entries)
{
  for (int i = 0; i < MAX_NUM_ENTRIES; ++i) {
    free_hash_entry(&entries[i]);
  }
}

void generate_key(char (*p_block_names)[32],
                  const int* const block_ids,
                  const int block_level,
                  char* const p_key)
{
  int total_len = 0;
  for (int i = 0; i < block_level; ++i) {
    int len = sprintf(p_key + total_len, "%s%02d/",
                      p_block_names[i], block_ids[i]);
    total_len += len;
  }
  sprintf(p_key + total_len, "%s%02d",
          p_block_names[block_level], block_ids[block_level]);
}

#define KEY_WORD 1
#define VAL_WORD 2

void parse_prototxt(const char* const filename)
{
  FILE* fp = fopen(filename, "r");

  char a_buf[32];
  char a_block_names[10][32];
  int block_ids[10] = { 0, };
  int block_level = 0;
  char a_key[4096];
  int word_type = KEY_WORD;
  int len = 0;

  HashEntry* p_entry = NULL;
  HashEntry* p_entry_parent = NULL;
  HashEntry* p_entry_root = find_or_make_hash_entry(
      gs_all_entries, "__root__", "__root__", ENTRY_VAL);

  while (!feof(fp)) {
    pop_spaces(fp);
    len = read_word(fp, a_buf);

    if (a_buf[0] == '{') {
      ++block_ids[block_level];
      generate_key(a_block_names, block_ids, block_level, a_key);
      p_entry = find_or_make_hash_entry(
          gs_all_entries, a_block_names[block_level], a_key, ENTRY_VAL);

      if (block_level == 0) {
        append_value_to_hash_entry(p_entry_root, (void*)p_entry);
      }
      else {
        generate_key(a_block_names, block_ids, block_level - 1, a_key);
        p_entry_parent = find_hash_entry(gs_all_entries, a_key, ENTRY_VAL);
        if (p_entry_parent) {
          append_value_to_hash_entry(p_entry_parent, (void*)p_entry);
        }
        else {
          printf("[ERROR] No parent for Entry %s at block level %d",
                 p_entry->p_key, block_level);
        }
      }

      ++block_level;
      block_ids[block_level] = 0;
      word_type = KEY_WORD;
    }
    else if (a_buf[0] == '}') {
      word_type = KEY_WORD;
      --block_level;
    }
    else if (a_buf[0] == ':') {
      word_type = VAL_WORD;
    }
    else if (word_type == KEY_WORD) {
      strcpy(a_block_names[block_level], a_buf);
    }

    else {
      void* p_value = (void*)malloc(strlen(a_buf) * sizeof(char));
      strcpy(p_value, a_buf);

      generate_key(a_block_names, block_ids, block_level, a_key);
      p_entry = find_or_make_hash_entry(
          gs_all_entries, a_block_names[block_level], a_key, STRING_VAL);
      append_value_to_hash_entry(p_entry, p_value);

      if (p_entry->num_values == 1) {
        if (block_level == 0) {
          append_value_to_hash_entry(p_entry_root, (void*)p_entry);
        }
        else {
          generate_key(a_block_names, block_ids, block_level - 1, a_key);
          p_entry_parent = find_hash_entry(gs_all_entries, a_key, ENTRY_VAL);
          if (p_entry_parent) {
            append_value_to_hash_entry(p_entry_parent, (void*)p_entry);
          }
          else {
            printf("[ERROR] No parent for Entry %s at block level %d",
                   p_entry->p_key, block_level);
          }
        }
      }

      word_type = KEY_WORD;
    }
  }
}

int main(int argc, char* argv[])
{
  parse_prototxt(argv[1]);

  HashEntry* p_entry_root =
      find_hash_entry(gs_all_entries, "__root__", ENTRY_VAL);

  print_hash_entry(p_entry_root, 0);

  free_hash_table(gs_all_entries);
  return 0;
}
