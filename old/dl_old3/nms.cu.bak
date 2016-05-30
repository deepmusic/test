#include "layer.h"

// --------------------------------------------------------------------------
// kernel code
//   iou: compute overlap between two boxes
//   nms_mask: given a set of boxes, compute overlap between all box pairs
// --------------------------------------------------------------------------

// "IoU = intersection area / union area" of two boxes A, B
//   A, B: 4-dim array (x1, y1, x2, y2)
#ifdef GPU
__device__
#endif
real iou(const real* const A, const real* const B)
{
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
typedef unsigned long long uint64;
#define NMS_BLOCK_SIZE 64
#ifdef GPU
__global__
void nms_mask_gpu(const real* const boxes,
                  uint64* const mask,
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
void nms_mask_cpu(const real* const boxes,
                  uint64* const mask,
                  const int num_boxes, const real nms_thresh)
{
  // number of blocks along each dimension
  const int num_blocks = DIV_THEN_CEIL(num_boxes,  NMS_BLOCK_SIZE);

  // the whole 2-dim computations "num_boxes x num_boxes" is done by
  // sweeping all "64 x 64"-sized blocks
  for (int j_start = 0; j_start < num_boxes; j_start += NMS_BLOCK_SIZE) {
    for (int i_start = 0; i_start < num_boxes; i_start += NMS_BLOCK_SIZE) {
      // block region
      //   j = j_start + { 0, ..., dj_end - 1 }
      //   i = i_start + { 0, ..., di_end - 1 }
      const int di_end = MIN(num_boxes - i_start,  NMS_BLOCK_SIZE);
      const int dj_end = MIN(num_boxes - j_start,  NMS_BLOCK_SIZE);

      // check whether box i is significantly overlapped with box j
      // for all j = j_start + { 0, ..., dj_end - 1 },
      //         i = i_start + { 0, ..., di_end - 1 },
      // except for i == j
      for (int dj = 0; dj < dj_end; ++dj) {
        // box j & overlap mask
        const real* const box_j = boxes + (j_start + dj) * 5;
        uint64 mask_j = 0;

        // check for all i = i_start + { 0, ..., di_end - 1 }
        // except for i == j
        const int di_start = (i_start == j_start) ? (dj + 1) : 0;
        for (int di = di_start; di < di_end; ++di) {
          // box i
          const real* const box_i = boxes + (i_start + di) * 5;

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
          const int bi = i_start / NMS_BLOCK_SIZE;
          mask[(j_start + dj) * num_blocks + bi] = mask_j;
        }
      } // endfor dj
    } // endfor j_start
  } // endfor i_start
}
#endif



// --------------------------------------------------------------------------
// operator code
//   nms: given a set of boxes, discard significantly-overlapped boxes
// --------------------------------------------------------------------------

// given box proposals (sorted in descending order of their scores),
// discard a box if it is significantly overlapped with
// one or more previous (= scored higher) boxes
//   num_boxes: number of box proposals given
//   boxes: "num_boxes x 5" array (x1, y1, x2, y2, score)
//          sorted in descending order of scores
//   num_out: number of remaining boxes
//   keep_out: "num_out x 1" array
//             indices of remaining boxes
//   base_index: a constant added to keep_out,  usually 0
//               keep_out[i] = base_index + actual index in boxes
//   nms_thresh: threshold for determining "significant overlap"
//               if "intersection area / union area > nms_thresh",
//               two boxes are thought of as significantly overlapped
void nms(const int num_boxes, const real* const boxes,
         int* const num_out, int* const keep_out, const int base_index,
         const real nms_thresh, const int max_num_out)
{
  const int num_blocks = DIV_THEN_CEIL(num_boxes,  NMS_BLOCK_SIZE);
  uint64* const mask
      = (uint64*)malloc(num_boxes * num_blocks * sizeof(uint64));

  #ifdef GPU
  {
    real* boxes_dev;
    uint64* mask_dev;
    const dim3 blocks(num_blocks, num_blocks);

    // GPU memory allocation & copy box data
    cudaMalloc(&boxes_dev, num_boxes * 5 * sizeof(real));
    cudaMemcpyAsync(boxes_dev, boxes, num_boxes * 5 * sizeof(real),
                    cudaMemcpyHostToDevice);
    cudaMalloc(&mask_dev, num_boxes * num_blocks * sizeof(uint64));

    // find all significantly-overlapped pairs of boxes
    nms_mask_gpu<<<blocks, NMS_BLOCK_SIZE>>>(
        boxes_dev,  mask_dev,  num_boxes,  nms_thresh);

    // copy mask data to main memory
    cudaMemcpyAsync(mask, mask_dev, sizeof(uint64) * num_boxes * num_blocks,
                    cudaMemcpyDeviceToHost);

    // GPU memory deallocation
    cudaFree(boxes_dev);
    cudaFree(mask_dev);
  }
  #else
  {
    // find all significantly-overlapped pairs of boxes
    nms_mask_cpu(boxes,  mask,  num_boxes,  nms_thresh);
  }
  #endif

  // discard i-th box if it is significantly overlapped with
  // one or more previous (= scored higher) boxes
  {
    int num_to_keep = 0;
    uint64* const remv = (uint64*)calloc(num_blocks, sizeof(uint64));

    for (int i = 0; i < num_boxes; ++i) {
      const int nblock = i / NMS_BLOCK_SIZE;
      const int inblock = i % NMS_BLOCK_SIZE;

      if (!(remv[nblock] & (1ULL << inblock))) {
        keep_out[num_to_keep++] = base_index + i;
        uint64* p = mask + i * num_blocks;
        for (int j = nblock; j < num_blocks; ++j) {
          remv[j] |= p[j];
        }

        if (num_to_keep == max_num_out) {
          break;
        }
      }
    }
    *num_out = num_to_keep;

    free(remv);
  }

  free(mask);
}
