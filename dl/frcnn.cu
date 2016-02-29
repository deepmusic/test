typedef struct PVANET_
{
  Tensor input;
  Tensor conv1_1, conv1_2;
  Tensor weight1_1, bias1_1, weight1_2, bias1_2;
  Tensor conv2_1, conv2_2;
  Tensor weight2_1, bias2_1, weight2_2, bias2_2;
  Tensor conv3_1, conv3_2, conv3_3;
  Tensor weight3_1, bias3_1, weight3_2, bias3_2, weight3_3, bias3_3;
  Tensor conv4_1, conv4_2, conv4_3;
  Tensor weight4_1, bias4_1, weight4_2, bias4_2, weight4_3, bias4_3;
  Tensor conv5_1, conv5_2, conv5_3;
  Tensor weight5_1, bias5_1, weight5_2, bias5_2, weight5_3, bias5_3;
  Tensor downsample, upsample, concat;
  Tensor weight_up;
  Tensor convf;
  Tensor weightf, biasf;
} PVANET;

typedef struct SRPN_
{
  Tensor conv1, conv3, conv5;
  Tensor score1, score3, score3;
  Tensor bbox1, bbox3, bbox5;
  Tensor score, bbox;
  Tensor roi;
} SRPN;

typedef struct RCNN_
{
  Tensor input;
  Tensor fc6;
  Tensor fc7;
  Tensor score, bbox;
} RCNN;

PVANET pvanet;
SRPN srpn;
RCNN rcnn;

void test_frcnn_7_1_1(void)
{
  conv_forward(&pvanet.input, &pvanet.conv1_1,
               temp_data, const_data, &option);
}
