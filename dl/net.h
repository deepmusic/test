typedef struct Conv_
{
  Tensor bottom;
  Tensor top;
  Tensor weight;
  Tensor bias;
  real* temp_data;
  real* const_data;
  ConvOption option;
} Conv;

Conv* conv_init(const ConvOption* const option);

void conv_forward(Conv* const conv);

inline void conv_forward(Conv* const conv)
{
  conv_forward(&conv->bottom3d, &conv->top3d, &conv->weight5d, &conv->bias1d,
               conv->temp_data, conv->const_data, &conv->option);
}

