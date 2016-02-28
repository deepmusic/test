
    unsigned int* mask =
        static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
    caffe_gpu_rng_uniform(count, mask);
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), count, mask));
  CUDA_KERNEL_LOOP(index, count) {
    top[index] = bottom[index] * (mask[index] > threshold) / (1.0f - threshold);
  }
