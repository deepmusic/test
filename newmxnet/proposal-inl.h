/*!
 * Copyright (c) 2016 by Kye-Hyeon Kim
 * \file proposal-inl.h
 * \brief proposal operator and symbol
*/
#ifndef MXNET_OPERATOR_PROPOSAL_INL_H_
#define MXNET_OPERATOR_PROPOSAL_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"


namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace prps {
enum ProposalOpInputs {kData, kBox, kImginfo};
enum ProposalOpOutputs {kOut};
}  // prps

struct ProposalParam : public dmlc::Parameter<ProposalParam> {
  int feat_stride;
  int base_size;
  int pre_nms_topn;
  int post_nms_topn;
  float nms_thresh;
  int min_size;
  std::vector<float> ratios;
  TShape scales;
  DMLC_DECLARE_PARAMETER(ProposalParam) {
    // TODO(bing) change to only set lower bound
    // add support for boolean
    DMLC_DECLARE_FIELD(feat_stride).set_default(16)
    .describe("Proposal stride (raw image pixels).");
    DMLC_DECLARE_FIELD(base_size).set_default(16)
    .describe("Base anchor size.");
    DMLC_DECLARE_FIELD(pre_nms_topn).set_default(6000)
    .describe("Number of candidates to be generated before NMS operation.");
    DMLC_DECLARE_FIELD(post_nms_topn).set_default(300)
    .describe("Number of final proposals to be chosen by NMS operation.");
    DMLC_DECLARE_FIELD(nms_thresh).set_default(0.7)
    .describe("Minimum area of overlap for merging two regions.");
    DMLC_DECLARE_FIELD(min_size).set_default(16)
    .describe("Minimum proposal height and width (raw image pixels).");
    DMLC_DECLARE_FIELD(scales)
    .set_expect_ndim(5).enforce_nonzero()
    .describe("anchor scales (* base anchor size): (s1, s2, s3, s4, s5)");
  }
};

template<typename Dtype>
class BoundingBox {
 public:
  Dtype x1, y1, x2, y2;
  Dtype score;
  int class_idx;

  BoundingBox() {
    this->x1 = 0; this->y1 = 0; this->x2 = 0; this->y2 = 0;
    this->score = 0;
    this->class_idx = -1;
  }
  BoundingBox(Dtype x1, Dtype y1, Dtype x2, Dtype y2) {
    this->x1 = x1; this->y1 = y1; this->x2 = x2; this->y2 = y2;
    this->score = 0;
    this->class_idx = -1;
  }
  BoundingBox(Dtype x1, Dtype y1, Dtype x2, Dtype y2, Dtype score) {
    this->x1 = x1; this->y1 = y1; this->x2 = x2; this->y2 = y2;
    this->score = score;
    this->class_idx = -1;
  }

  bool operator<(BoundingBox other) const { return score > other.score; }

  bool transform_box(Dtype dx, Dtype dy, Dtype dw, Dtype dh, int im_w, int im_h, Dtype min_w, Dtype min_h) {
    Dtype w = x2 - x1 + 1.0f;
    Dtype h = y2 - y1 + 1.0f;
    Dtype ctr_x = x1 + 0.5f * w;
    Dtype ctr_y = y1 + 0.5f * h;

    Dtype pred_ctr_x = dx * w + ctr_x;
    Dtype pred_ctr_y = dy * h + ctr_y;
    Dtype pred_w = exp(dw) * w;
    Dtype pred_h = exp(dh) * h;

    x1 = pred_ctr_x - 0.5f * pred_w;
    y1 = pred_ctr_y - 0.5f * pred_h;
    x2 = pred_ctr_x + 0.5f * pred_w;
    y2 = pred_ctr_y + 0.5f * pred_h;

    x1 = std::max<Dtype>(std::min<Dtype>(x1, im_w - 1), 0);
    y1 = std::max<Dtype>(std::min<Dtype>(y1, im_h - 1), 0);
    x2 = std::max<Dtype>(std::min<Dtype>(x2, im_w - 1), 0);
    y2 = std::max<Dtype>(std::min<Dtype>(y2, im_h - 1), 0);

    w = x2 - x1 + 1.0f;
    h = y2 - y1 + 1.0f;

    if (w >= min_w && h >= min_h) return true;
    return false;
  }
};

/**
 * \brief This is the implementation of proposal operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu>
class ProposalOp : public Operator {
 public:
  explicit ProposalOp(ProposalParam p) {
    this->param_ = p;
    param_.ratios.push_back(0.5f);
    param_.ratios.push_back(0.75f);
    param_.ratios.push_back(1.0f);
    param_.ratios.push_back(1.5f);
    param_.ratios.push_back(2.0f);

    // generate anchors
    int base_anchor[4] = { 0, 0, param_.base_size - 1, param_.base_size - 1 };
    float w, h, x_ctr, y_ctr;
    w = base_anchor[2] - base_anchor[0] + 1.0f;
    h = base_anchor[3] - base_anchor[1] + 1.0f;
    x_ctr = base_anchor[0] + 0.5f * (w - 1);
    y_ctr = base_anchor[1] + 0.5f * (h - 1);
    std::vector<float> ratio_anchors(param_.ratios.size());
    for (int i = 0; i < param_.ratios.size(); ++i) {
      float ws, hs;
      ws = 0.5f * (round(sqrt(w * h / param_.ratios[i])) - 1);
      hs = 0.5f * (round(ws * param_.ratios[i]) - 1);
      ratio_anchors.push_back(x_ctr - ws);
      ratio_anchors.push_back(y_ctr - hs);
      ratio_anchors.push_back(x_ctr + ws);
      ratio_anchors.push_back(y_ctr + hs);
    }
    for (int i = 0; i < ratio_anchors.size(); i += 4) {
      const float *ratio_anchor = &ratio_anchors[i];
      w = ratio_anchor[2] - ratio_anchor[0] + 1;
      h = ratio_anchor[3] - ratio_anchor[1] + 1;
      x_ctr = ratio_anchor[0] + 0.5f * (w - 1);
      y_ctr = ratio_anchor[1] + 0.5f * (h - 1);
      for (int i = 0; i < param_.scales.ndim(); ++i) {
        float ws, hs;
        ws = 0.5f * (w * param_.scales[i] - 1);
        hs = 0.5f * (h * param_.scales[i] - 1);
        anchors_.push_back(x_ctr - ws);
        anchors_.push_back(y_ctr - hs);
        anchors_.push_back(x_ctr + ws);
        anchors_.push_back(y_ctr + hs);
      }
    }
    num_anchors_ = anchors_.size() / 4;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    if (req[prps::kOut] == kNullOp) return;
    CHECK_EQ(req[prps::kOut], kWriteTo);
    size_t expected = 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(in_data[prps::kData].size(0), 1) << "Only single item batches are supported.";
    Stream<xpu> *s1 = ctx.get_stream<xpu>();
    Stream<xpu> *s2 = ctx.get_stream<xpu>();

    Tensor<xpu, 4> data_xpu = in_data[prps::kData].get<xpu, 4, real_t>(s1);
    Tensor<xpu, 4> bbox_xpu = in_data[prps::kBox].get<xpu, 4, real_t>(s2);
    Tensor<xpu, 2> img_info_xpu = in_data[prps::kImginfo].get<xpu, 2, real_t>(s2);
    Tensor<xpu, 2> roi_xpu = out_data[prps::kOut].get<xpu, 2, real_t>(s1);

    Tensor<cpu, 4> data(data_xpu.shape_); AllocSpace(&data); Copy(data, data_xpu, s1);
    Tensor<cpu, 4> bbox(bbox_xpu.shape_); AllocSpace(&bbox); Copy(bbox, bbox_xpu, s2);
    Tensor<cpu, 2> img_info(img_info_xpu.shape_); AllocSpace(&img_info); Copy(img_info, img_info_xpu, s2);
    Tensor<cpu, 2> roi(roi_xpu.shape_); AllocSpace(&roi);
    img_info[0][0] = 640;
    img_info[0][1] = 640;
    img_info[0][2] = 1.0f;
    img_info[0][3] = 1.0f;

    int height = in_data[prps::kData].shape_[2];
    int width = in_data[prps::kData].shape_[3];
    std::vector<BoundingBox<real_t> > proposals;
    for (int k = 0; k < num_anchors_; ++k) {
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
          real_t x1 = j * param_.feat_stride + anchors_[k * 4 + 0];
          real_t y1 = i * param_.feat_stride + anchors_[k * 4 + 1];
          real_t x2 = j * param_.feat_stride + anchors_[k * 4 + 2];
          real_t y2 = i * param_.feat_stride + anchors_[k * 4 + 3];

          real_t dx = bbox[0][k * 4 + 0][i][j];
          real_t dy = bbox[0][k * 4 + 1][i][j];
          real_t dw = bbox[0][k * 4 + 2][i][j];
          real_t dh = bbox[0][k * 4 + 3][i][j];
          real_t score = data[0][num_anchors_ + k][i][j];

          BoundingBox<real_t> proposal(x1, y1, x2, y2, score);
          bool box_created = proposal.transform_box(dx, dy, dw, dh,
                                                    img_info[0][1], img_info[0][0],
                                                    param_.min_size * img_info[0][2],
                                                    param_.min_size * img_info[0][3]);
          if (box_created) proposals.push_back(proposal);
        }
      }
    }
    std::sort(proposals.begin(), proposals.end());

    if (param_.pre_nms_topn > 0) {
      while (proposals.size() > param_.pre_nms_topn) proposals.pop_back();
    }

    real_t *sorted_dets = (real_t*)calloc(proposals.size() * 5, sizeof(real_t));
    for (int i = 0; i < proposals.size(); ++i) {
      sorted_dets[i * 5 + 0] = proposals[i].x1;
      sorted_dets[i * 5 + 1] = proposals[i].y1;
      sorted_dets[i * 5 + 2] = proposals[i].x2;
      sorted_dets[i * 5 + 3] = proposals[i].y2;
      sorted_dets[i * 5 + 4] = proposals[i].score;
    }
    int *keep = (int*)calloc(proposals.size(), sizeof(int));
    int num_out = 0;
    Tensor<xpu, 1, real_t> dummy;
    _nms(keep, &num_out, sorted_dets, proposals.size(), 5, param_.nms_thresh, dummy);
    free(sorted_dets);

    int nproposals = std::min<int>(num_out, param_.post_nms_topn);
    for (int i = 0; i < nproposals; ++i) {
      roi[i][0] = 0;
      roi[i][1] = proposals[keep[i]].x1;
      roi[i][2] = proposals[keep[i]].y1;
      roi[i][3] = proposals[keep[i]].x2;
      roi[i][4] = proposals[keep[i]].y2;
      std::cout << "roi " << i << ": (" << roi[i][1] << "," << roi[i][2] << "), (" << roi[i][3] << "," << roi[i][4] << ")" << std::endl;
    }
    for (int i = nproposals; i < param_.post_nms_topn; ++i) {
      roi[i][0] = -1;
      roi[i][1] = 0;
      roi[i][2] = 0;
      roi[i][3] = 0;
      roi[i][4] = 0;
    }
    free(keep);

    Copy(roi_xpu, roi, s1);

    FreeSpace(&data);
    FreeSpace(&bbox);
    FreeSpace(&img_info);
    FreeSpace(&roi);
  }

 private:
  ProposalParam param_;
  std::vector<float> anchors_;
  int num_anchors_;
};  // class ProposalOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(ProposalParam param);

#if DMLC_USE_CXX11
class ProposalProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "bbox", "img_info"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Input:[data, bbox, img_info]";

    // data: [1, 2*num_anchors, h, w]
    TShape dshape = in_shape->at(prps::kData);
    CHECK_EQ(dshape.ndim(), 4) << "data should be a 4D tensor";
    CHECK_EQ(dshape[0], 1) << "Only single item batches are supported.";
    size_t num_anchors = 5 * param_.scales.ndim();
    CHECK_EQ(dshape[1], 2 * num_anchors)
        << "Shape mismatch: data[1, "
        << dshape[1] << ", h, w] != 2 * " << num_anchors;

    // bbox: [1, 4*num_anchors, h, w]
    TShape bshape = in_shape->at(prps::kBox);
    CHECK_EQ(bshape.ndim(), 4) << "bbox should be a 4D tensor";
    CHECK_EQ(bshape[0], 1) << "Only single item batches are supported.";
    CHECK_EQ(bshape[1], 4 * num_anchors)
        << "Shape mismatch: bbox[1, "
        << bshape[1] << ", h, w] != 4 * " << num_anchors;
    CHECK_EQ(dshape[2], bshape[2])
        << "Shape mismatch: data[1, 2*anchors, "
        << dshape[2] << ", " << dshape[3]
        << "] != bbox[1, 4*anchors, "
        << bshape[2] << ", " << bshape[3] << "]";
    CHECK_EQ(dshape[3], bshape[3])
        << "Shape mismatch: data[1, 2*anchors, "
        << dshape[2] << ", " << dshape[3]
        << "] != bbox[1, 4*anchors, "
        << bshape[2] << ", " << bshape[3] << "]";

    // img_info: [1, 4]
    TShape ishape = in_shape->at(prps::kImginfo);
    SHAPE_ASSIGN_CHECK(*in_shape, prps::kImginfo, Shape2(1, 4));
    //CHECK_EQ(ishape.ndim(), 2) << "img_info should be a 2D tensor of shape [1, 4]";
    //CHECK_EQ(ishape[0], 1) << "img_info should be a 2D tensor of shape [1, 4]";
    //CHECK_EQ(ishape[1], 4) << "img_info should be a 2D tensor of shape [1, 4]";

    // out: [num_proposals, 5]
    out_shape->clear();
    out_shape->push_back(Shape2(param_.post_nms_topn, 5));
    return true;
  }

  OperatorProperty* Copy() const override {
    ProposalProp* proposal_sym = new ProposalProp();
    proposal_sym->param_ = this->param_;
    return proposal_sym;
  }

  std::string TypeString() const override {
    return "Proposal";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[prps::kOut], in_data[prps::kData], in_data[prps::kBox]};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  ProposalParam param_;
};  // class ProposalSymbol
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_PROPOSAL_INL_H_
