#include "layers/nms.h"
#include <string.h>

// --------------------------------------------------------------------------
// kernel code
//   iou: compute size of overlap between two boxes
//   nms_mask: given a set of boxes, compute overlap between all box pairs
// --------------------------------------------------------------------------

// "IoU = intersection area / union area" of two boxes A, B
//   A, B: 4-dim array (x1, y1, x2, y2)
#ifdef GPU
__device__
#endif
static
real iou(const real A[], const real B[])
{
  #ifndef GPU
  if (A[0] > B[2] || A[1] > B[3] || A[2] < B[0] || A[3] < B[1]) {
    return 0;
  }
  else {
  #endif

  // overlapped region (= box)
  const real x1 = MAX(A[0],  B[0]);
  const real y1 = MAX(A[1],  B[1]);
  const real x2 = MIN(A[2],  B[2]);
  const real y2 = MIN(A[3],  B[3]);

  // intersection area
  const real width = MAX(0.0f,  x2 - x1 + 1.0f);
  const real height = MAX(0.0f,  y2 - y1 + 1.0f);
  const real area = width * height;

  // area of A, B
  const real A_area = (A[2] - A[0] + 1.0f) * (A[3] - A[1] + 1.0f);
  const real B_area = (B[2] - B[0] + 1.0f) * (B[3] - B[1] + 1.0f);

  // IoU
  return area / (A_area + B_area - area);

  #ifndef GPU
  }
  #endif
}

// given box proposals, compute overlap between all box pairs
// (overlap = intersection area / union area)
// and then set mask-bit to 1 if a pair is significantly overlapped
//   num_boxes: number of box proposals given
//   boxes: "num_boxes x 5" array (x1, y1, x2, y2, score)
//   nms_thresh: threshold for determining "significant overlap"
//               if "intersection area / union area > nms_thresh",
//               two boxes are thought of as significantly overlapped
// the all-pair computation (num_boxes x num_boxes) is done by
// divide-and-conquer:
//   each GPU block (bj, bi) computes for "64 x 64" box pairs (j, i),
//     j = bj * 64 + { 0, 1, ..., 63 }
//     i = bi * 64 + { 0, 1, ..., 63 },
//   and each "1 x 64" results is saved into a 64-bit mask
//     mask: "num_boxes x num_blocks" array
//     for mask[j][bi], "di-th bit = 1" means:
//       box j is significantly overlapped with box i,
//       where i = bi * 64 + di
#ifdef GPU
#define NMS_BLOCK_SIZE 64
typedef unsigned long long uint64;
__global__
static
void nms_mask_gpu(const real boxes[], uint64 mask[],
                  const int num_boxes, const real nms_thresh)
{
  // block region
  //   j = j_start + { 0, ..., dj_end - 1 }
  //   i = i_start + { 0, ..., di_end - 1 }
  const int i_start = blockIdx.x * NMS_BLOCK_SIZE;
  const int di_end = MIN(num_boxes - i_start,  NMS_BLOCK_SIZE);
  const int j_start = blockIdx.y * NMS_BLOCK_SIZE;
  const int dj_end = MIN(num_boxes - j_start,  NMS_BLOCK_SIZE);

  // copy all i-th boxes to GPU cache
  //   i = i_start + { 0, ..., di_end - 1 }
  __shared__ real boxes_i[NMS_BLOCK_SIZE * 4];
  {
    const int di = threadIdx.x;
    if (di < di_end) {
      boxes_i[di * 4 + 0] = boxes[(i_start + di) * 5 + 0];
      boxes_i[di * 4 + 1] = boxes[(i_start + di) * 5 + 1];
      boxes_i[di * 4 + 2] = boxes[(i_start + di) * 5 + 2];
      boxes_i[di * 4 + 3] = boxes[(i_start + di) * 5 + 3];
    }
  }
  __syncthreads();

  // given j = j_start + dj,
  //   check whether box i is significantly overlapped with box j
  //   (i.e., IoU(box j, box i) > threshold)
  //   for all i = i_start + { 0, ..., di_end - 1 } except for i == j
  {
    const int dj = threadIdx.x;
    if (dj < dj_end) {
      // box j
      const real* const box_j = boxes + (j_start + dj) * 5;

      // mask for significant overlap
      //   if IoU(box j, box i) > threshold,  di-th bit = 1
      uint64 mask_j = 0;

      // check for all i = i_start + { 0, ..., di_end - 1 }
      // except for i == j
      const int di_start = (i_start == j_start) ? (dj + 1) : 0;
      for (int di = di_start; di < di_end; ++di) {
        // box i
        const real* const box_i = boxes_i + di * 4;

        // if IoU(box j, box i) > threshold,  di-th bit = 1
        if (iou(box_j, box_i) > nms_thresh) {
          mask_j |= 1ULL << di;
        }
      }

      // mask: "num_boxes x num_blocks" array
      //   for mask[j][bi], "di-th bit = 1" means:
      //     box j is significantly overlapped with box i = i_start + di,
      //     where i_start = bi * block_size
      {
        const int num_blocks = DIV_THEN_CEIL(num_boxes,  NMS_BLOCK_SIZE);
        const int bi = blockIdx.x;
        mask[(j_start + dj) * num_blocks + bi] = mask_j;
      }
    } // endif dj < dj_end
  }
}
#else
#endif



// --------------------------------------------------------------------------
// auxiliary data structure
// --------------------------------------------------------------------------

// auxiliary data structure for NMS operation
#ifdef GPU
typedef struct NMSAuxData_
{
  uint64* mask_cpu;
  uint64* dead_bit_cpu;
  uint64* mask_gpu;
} NMSAuxData;
#else
typedef struct NMSAuxData_
{
  unsigned char* is_dead;
} NMSAuxData;
#endif

// auxiliary data initializer
#ifdef GPU
static
void malloc_nms_aux_data_gpu(NMSAuxData* const aux_data,
                             const int num_boxes,
                             long int* const p_space_cpu,
                             long int* const p_space_gpu)
{
  const int num_blocks = DIV_THEN_CEIL(num_boxes,  NMS_BLOCK_SIZE);

  aux_data->mask_cpu
      = (uint64*)calloc(num_boxes * num_blocks, sizeof(uint64));
  aux_data->dead_bit_cpu = (uint64*)calloc(num_blocks, sizeof(uint64));

  cudaMalloc(&aux_data->mask_gpu, num_boxes * num_blocks * sizeof(uint64));
  cudaMemset(aux_data->mask_gpu, 0, num_boxes * num_blocks * sizeof(uint64));

  *p_space_cpu = num_boxes * num_blocks * sizeof(uint64)
                 + num_blocks * sizeof(uint64);
  *p_space_gpu = num_boxes * num_blocks * sizeof(uint64);
}
#else
static
void malloc_nms_aux_data_cpu(NMSAuxData* const aux_data,
                             const int num_boxes,
                             long int* const p_space_cpu,
                             long int* const p_space_gpu)
{
  aux_data->is_dead
      = (unsigned char*)calloc(num_boxes, sizeof(unsigned char));

  *p_space_cpu = num_boxes * sizeof(unsigned char);
  *p_space_gpu = 0;
}
#endif

// auxiliary data finalizer
#ifdef GPU
static
void free_nms_aux_data_gpu(NMSAuxData* const aux_data)
{
  free(aux_data->mask_cpu);
  free(aux_data->dead_bit_cpu);
  cudaFree(aux_data->mask_gpu);

  memset(aux_data, 0, sizeof(NMSAuxData));
}
#else
static
void free_nms_aux_data_cpu(NMSAuxData* const aux_data)
{
  free(aux_data->is_dead);

  memset(aux_data, 0, sizeof(NMSAuxData));
}
#endif



// --------------------------------------------------------------------------
// operator code
// --------------------------------------------------------------------------

// given box proposals (sorted in descending order of their scores),
// discard a box if it is significantly overlapped with
// one or more previous (= scored higher) boxes
//   num_boxes: number of box proposals given
//   boxes: "num_boxes x 5" array (x1, y1, x2, y2, score)
//          sorted in descending order of scores
//   aux_data: auxiliary data for NMS operation
//   num_out: number of remaining boxes
//   index_out_cpu: "num_out x 1" array
//                  indices of remaining boxes
//                  allocated at main memory
//   base_index: a constant added to index_out_cpu,  usually 0
//               index_out_cpu[i] = base_index + actual index in boxes
//   nms_thresh: threshold for determining "significant overlap"
//               if "intersection area / union area > nms_thresh",
//               two boxes are thought of as significantly overlapped
//   bbox_vote: whether bounding-box voting is used (= 1) or not (= 0)
//   vote_thresh: threshold for selecting overlapped boxes
//                which are participated in bounding-box voting
#ifdef GPU
void nms(const int num_boxes, real boxes[], void* const aux_data,
         int* const num_out, int index_out_cpu[],
         const int base_index, const real nms_thresh, const int max_num_out,
         const int bbox_vote, const real vote_thresh)
{
  const int num_blocks = DIV_THEN_CEIL(num_boxes,  NMS_BLOCK_SIZE);
  uint64* const mask_cpu = ((NMSAuxData*)aux_data)->mask_cpu;

  {
    uint64* const mask_gpu = ((NMSAuxData*)aux_data)->mask_gpu;
    const dim3 blocks(num_blocks, num_blocks);

    // find all significantly-overlapped pairs of boxes
    nms_mask_gpu<<<blocks, NMS_BLOCK_SIZE>>>(
        boxes,  mask_gpu,  num_boxes,  nms_thresh);

    cudaMemcpyAsync(mask_cpu, mask_gpu,
                    sizeof(uint64) * num_boxes * num_blocks,
                    cudaMemcpyDeviceToHost);
  }

  // discard i-th box if it is significantly overlapped with
  // one or more previous (= scored higher) boxes
  {
    int num_selected = 0;
    uint64* const dead_bit_cpu = ((NMSAuxData*)aux_data)->dead_bit_cpu;

    memset(dead_bit_cpu, 0, num_blocks * sizeof(uint64));

    for (int i = 0; i < num_boxes; ++i) {
      const int nblock = i / NMS_BLOCK_SIZE;
      const int inblock = i % NMS_BLOCK_SIZE;

      if (!(dead_bit_cpu[nblock] & (1ULL << inblock))) {
        index_out_cpu[num_selected++] = base_index + i;
        const uint64* const mask_i = mask_cpu + i * num_blocks;
        for (int j = nblock; j < num_blocks; ++j) {
          dead_bit_cpu[j] |= mask_i[j];
        }

        if (num_selected == max_num_out) {
          break;
        }
      }
    }
    *num_out = num_selected;
  }
}
#else
void nms(const int num_boxes, real boxes[], void* const aux_data,
         int* const num_out, int index_out_cpu[],
         const int base_index, const real nms_thresh, const int max_num_out,
         const int bbox_vote, const real vote_thresh)
{
  unsigned char* const is_dead = ((NMSAuxData*)aux_data)->is_dead;
  int num_selected = 0;

  memset(is_dead, 0, num_boxes * sizeof(unsigned char));

  for (int i = 0; i < num_boxes; ++i) {
    if (is_dead[i]) {
      continue;
    }

    index_out_cpu[num_selected++] = base_index + i;

    if (bbox_vote) {
      real sum_score = boxes[i * 5 + 4];
      real sum_box[4] = {
          sum_score * boxes[i * 5 + 0], sum_score * boxes[i * 5 + 1],
          sum_score * boxes[i * 5 + 2], sum_score * boxes[i * 5 + 3]
      };

      for (int j = 0; j < i; ++j) {
        if (is_dead[j] && iou(&boxes[i * 5], &boxes[j * 5]) > vote_thresh) {
          real score = boxes[j * 5 + 4];
          sum_box[0] += score * boxes[j * 5 + 0];
          sum_box[1] += score * boxes[j * 5 + 1];
          sum_box[2] += score * boxes[j * 5 + 2];
          sum_box[3] += score * boxes[j * 5 + 3];
          sum_score += score;
        }
      }
      for (int j = i + 1; j < num_boxes; ++j) {
        real iou_val = iou(&boxes[i * 5], &boxes[j * 5]);
        if (!is_dead[j] && iou_val > nms_thresh) {
          is_dead[j] = 1;
        }
        if (iou_val > vote_thresh) {
          real score = boxes[j * 5 + 4];
          sum_box[0] += score * boxes[j * 5 + 0];
          sum_box[1] += score * boxes[j * 5 + 1];
          sum_box[2] += score * boxes[j * 5 + 2];
          sum_box[3] += score * boxes[j * 5 + 3];
          sum_score += score;
        }
      }

      boxes[i * 5 + 0] = sum_box[0] / sum_score;
      boxes[i * 5 + 1] = sum_box[1] / sum_score;
      boxes[i * 5 + 2] = sum_box[2] / sum_score;
      boxes[i * 5 + 3] = sum_box[3] / sum_score;
    }

    else {
      for (int j = i + 1; j < num_boxes; ++j) {
        if (!is_dead[j] && iou(&boxes[i * 5], &boxes[j * 5]) > nms_thresh) {
          is_dead[j] = 1;
        }
      }
    }

    if (num_selected == max_num_out) {
      break;
    }
  }

  *num_out = num_selected;
}
#endif



// --------------------------------------------------------------------------
// functions for layer-wise operators that use NMS operation
// --------------------------------------------------------------------------

void malloc_nms_aux_data(void** const p_aux_data,
                         int num_boxes,
                         long int* const p_space_cpu,
                         long int* const p_space_gpu)
{
  long int space_cpu, space_gpu;

  *p_aux_data = (void*)malloc(sizeof(NMSAuxData));

  #ifdef GPU
  malloc_nms_aux_data_gpu((NMSAuxData*)(*p_aux_data), num_boxes,
                          &space_cpu, &space_gpu);
  #else
  malloc_nms_aux_data_cpu((NMSAuxData*)(*p_aux_data), num_boxes,
                          &space_cpu, &space_gpu);
  #endif

  *p_space_cpu = space_cpu + sizeof(NMSAuxData);
  *p_space_gpu = space_gpu;
}

void free_nms_aux_data(void* const aux_data)
{
  #ifdef GPU
  free_nms_aux_data_gpu((NMSAuxData*)aux_data);
  #else
  free_nms_aux_data_cpu((NMSAuxData*)aux_data);
  #endif

  free(aux_data);
}
