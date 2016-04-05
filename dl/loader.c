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
  char* key;
  int type;
  int num_values;
  void** values;
} HashEntry;


static HashEntry gs_all_entries[MAX_NUM_ENTRIES] = {
  { 0, 0, 0, 0 },
};


unsigned int entry2hash(const char* const str,
                        const int type)
{
  const unsigned char* p_str = (unsigned char*)str;
  unsigned short hash = 5381;
  unsigned short ch;

  hash = ((hash << 5) + hash) + (unsigned short)type;
  // for ch = 0, ..., strlen(str)-1
  while (ch = *(p_str++)) {
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
                           const char* const key,
                           const int type)
{
  unsigned int hash = entry2hash(key, type) % MAX_NUM_ENTRIES;

  while (entries[hash].key) {
    if (strcmp(key, entries[hash].key) == 0) {
      return &entries[hash];
    }
    hash = (hash == MAX_NUM_ENTRIES - 1) ? 0 : hash + 1;
  }

  entries[hash].key = (char*)malloc((strlen(key) + 1) * sizeof(char));
  strcpy(entries[hash].key, key);
  entries[hash].type = type;
  entries[hash].values = NULL;
  entries[hash].num_values = 0;
  //printf("[INFO] Create Entry %s(%s)\n",
  //       entries[hash].key, gs_types[type]);

  return &entries[hash];
}

void set_hash_entry(HashEntry* const entries,
                    const char* const key,
                    const int type,
                    void* const value)
{
  HashEntry* entry = find_hash_entry(entries, key, type);

  if (!entry->values) {
    entry->values = (void**)malloc(sizeof(void*));
    entry->values[0] = value;
    entry->num_values = 1;
    //printf("[INFO] Memory allocation for Entry %s: %d values\n",
    //       entry->key, entry->num_values);
  }

  else if (is_two_to_the_n(entry->num_values)) {
    void** temp = (void**)calloc(entry->num_values * 2, sizeof(void*));
    memcpy(temp, entry->values, entry->num_values * sizeof(void*));
    free(entry->values);
    entry->values = temp;
    entry->values[entry->num_values++] = value;
    //printf("[INFO] Memory reallocation for Entry %s: %d values\n",
    //       entry->key, entry->num_values);
  }

  else {
    entry->values[entry->num_values++] = value;
    //printf("[INFO] New value for Entry %s: %d values\n",
    //       entry->key, entry->num_values);
  }
}

void print_hash_entry(const HashEntry* const entry, const int level)
{
  for (int i = 0; i < level; ++i) {
    printf("  ");
  }
  printf("Entry %s(%s): ", entry->key, gs_types[entry->type]);
  switch (entry->type) {
    case INT_VAL:
      printf("[");
      for (int i = 0; i < entry->num_values - 1; ++i) {
        printf("%d, ", *(((int**)entry->values)[i]));
      }
      printf("%d]\n", *(((int**)entry->values)[entry->num_values - 1]));
      break;
    case REAL_VAL:
      printf("[");
      for (int i = 0; i < entry->num_values - 1; ++i) {
        printf("%f, ", *(((real**)entry->values)[i]));
      }
      printf("%f]\n", *(((real**)entry->values)[entry->num_values - 1]));
      break;
    case STRING_VAL:
      printf("[");
      for (int i = 0; i < entry->num_values - 1; ++i) {
        printf("%s, ", ((const char**)entry->values)[i]);
      }
      printf("%s]\n", ((const char**)entry->values)[entry->num_values - 1]);
      break;
    case TENSOR_VAL:
      printf("[");
      for (int i = 0; i < entry->num_values - 1; ++i) {
        printf("%s, ", ((const Tensor**)entry->values)[i]->name);
      }
      printf("%s]\n", ((const Tensor**)entry->values)[entry->num_values - 1]->name);
      break;
    case ENTRY_VAL:
      for (int i = 0; i < entry->num_values - 1; ++i) {
        printf("\n");
        print_hash_entry(((const HashEntry**)entry->values)[i], level + 1);
      }
      break;
    case LAYER:
      printf("[");
      for (int i = 0; i < entry->num_values - 1; ++i) {
        printf("%s, ", ((const Layer**)entry->values)[i]->name);
      }
      printf("%s]\n", ((const Layer**)entry->values)[entry->num_values - 1]->name);
      break;
    case OPERATOR:
      printf("[");
      for (int i = 0; i < entry->num_values - 1; ++i) {
        printf("%s, ", ((const char**)entry->values)[i]);
      }
      printf("%s]\n", ((const char**)entry->values)[entry->num_values - 1]);
      break;
    default:
      printf("%d values of unknown types\n", entry->num_values);
      break;
  }
}

int str2type(const char* const word)
{
  {
    if (word[0] == '"') {
      const char* p_word = word;
      while (*(++p_word));
      if (*(p_word - 1) == '"') {
        return STRING_VAL;
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

int read_str(FILE* fp, char* const buf)
{
  char* p_buf = buf;
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

void free_hash_entry(HashEntry* entry)
{
  if (entry->key) {
    //print_hash_entry(entry, 0);
    free(entry->key);
  }
  if (entry->values) {
    if (entry->type != ENTRY_VAL) {
      for (int i = 0; i < entry->num_values; ++i) {
        if (entry->values[i]) {
          //printf("  free value %d\n", i);
          free(entry->values[i]);
        }
      }
    }
    free(entry->values);
  }
  memset(entry, 0, sizeof(HashEntry));
}

void free_hash_table(HashEntry* entries)
{
  for (int i = 0; i < MAX_NUM_ENTRIES; ++i) {
    free_hash_entry(&entries[i]);
  }
}

void generate_key(char (*keys)[32],
                  const int* const block_ids,
                  const int block_level,
                  char* const key)
{
  int total_len = 0;
  for (int i = 0; i < block_level; ++i) {
    int len = sprintf(key + total_len, "%s%02d/", keys[i], block_ids[i]);
    total_len += len;
  }
  sprintf(key + total_len, "%s", keys[block_level]);
}

#define KEY_WORD 1
#define VAL_WORD 2

int main(int argc, char* argv[])
{
  FILE* fp = fopen(argv[1], "r");

  char buf[32];
  char keys[10][32];
  int block_ids[10] = { 0, };
  int block_level = 0;
  char key_full[4096];
  int word_type = KEY_WORD;
  void* val = NULL;
  int len = 0;

  while (!feof(fp)) {
    pop_spaces(fp);
    len = read_str(fp, buf);

    if (buf[0] == '{') {
      ++block_ids[block_level];
      ++block_level;
      block_ids[block_level] = 0;
      word_type = KEY_WORD;

      generate_key(keys, block_ids, block_level, key_full);
      HashEntry* entry =
          find_hash_entry(gs_all_entries, key_full, ENTRY_VAL);

      generate_key(keys, block_ids, block_level - 1, key_full);
      set_hash_entry(gs_all_entries, key_full, ENTRY_VAL, entry);
      //printf("block opened, level = %d\n", block_level);
    }
    else if (buf[0] == '}') {
      word_type = KEY_WORD;
      --block_level;
      //printf("block closed, level = %d\n", block_level);
    }
    else if (buf[0] == ':') {
      word_type = VAL_WORD;
      //printf("change to value mode\n");
    }
    else if (word_type == KEY_WORD) {
      strcpy(keys[block_level], buf);
      //printf("load keyword %s\n", buf);
    }

    else {
      val = (void*)malloc(strlen(buf) * sizeof(char));
      strcpy(val, buf);
      generate_key(keys, block_ids, block_level, key_full);
      set_hash_entry(gs_all_entries, key_full, STRING_VAL, val);
      HashEntry* entry =
          find_hash_entry(gs_all_entries, key_full, STRING_VAL);

      word_type = KEY_WORD;

      print_hash_entry(entry, block_level);
    }
  }

  for (int i = 0; i < MAX_NUM_ENTRIES; ++i) {
    if (gs_all_entries[i].key) {
      print_hash_entry(&gs_all_entries[i], 0);
    }
  }

  free_hash_table(gs_all_entries);

  return 0;
}
