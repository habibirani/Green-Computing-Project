import torch.quantization as quant
import torch
from transformer import TimeSeriesTransformer

model = TimeSeriesTransformer()

# Define quantization configuration
quant_config = quant.get_default_qconfig('qnnpack')

# Apply post-training static quantization
quant_model = quant.quantize_dynamic(
    model, qconfig_spec={"": quant_config}, dtype=torch.qint8
)

# Apply dynamic quantization
quantized_model = quant.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)


# Define the quantization configuration
quant_config = quant.QConfig(activation=quant.MinMaxObserver.with_args(dtype=torch.qint8), weight=quant.MinMaxObserver.with_args(dtype=torch.qint8))

# Prepare the model for quantization-aware training
quantized_model = quant.prepare_qat(model, qconfig=quant_config)

# Train the quantized model
...

# Convert the trained model to a quantized model
quantized_model = quant.convert(model, inplace=False)




