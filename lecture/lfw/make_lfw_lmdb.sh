${CAFFE_ROOT}/build/tools/convert_imageset -encode_type='jpg' -encoded=true $1/lfw/ $2/train_1.txt $1/train_lmdb_1
${CAFFE_ROOT}/build/tools/convert_imageset -encode_type='jpg' -encoded=true $1/lfw/ $2/train_2.txt $1/train_lmdb_2
${CAFFE_ROOT}/build/tools/convert_imageset -encode_type='jpg' -encoded=true $1/lfw/ $2/test_1.txt $1/test_lmdb_1
${CAFFE_ROOT}/build/tools/convert_imageset -encode_type='jpg' -encoded=true $1/lfw/ $2/test_2.txt $1/test_lmdb_2
