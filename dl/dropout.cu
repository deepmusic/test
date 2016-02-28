
  if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT)
      != CURAND_STATUS_SUCCESS ||
      curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen())
      != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Curand generator. Curand won't be available.";
  }
    unsigned int* mask =
        static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
    caffe_gpu_rng_uniform(count, mask);
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), count, mask));
  CUDA_KERNEL_LOOP(index, count) {
    top[index] = bottom[index] * (mask[index] > threshold) / (1.0f - threshold);
  }
