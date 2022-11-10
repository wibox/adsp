import torch
import onnxruntime as ort
import numpy as np
from onnx2torch import convert

onnx_model_path = "concat_unet-v1.0.onnx"

torch_mdl = convert(onnx_model_path)

x = torch.ones((1, 2, 224, 224))

out_torch = torch_mdl(x)

ort_sess = ort.InferenceSession(onnx_model_path)
outputs_ort = ort_sess.run(None, {'input': x.numpy()})

# Check the Onnx output against PyTorch
print(torch.max(torch.abs(outputs_ort - out_torch.detach().numpy())))
print(np.allclose(outputs_ort, out_torch.detach().numpy(), atol=1.e-7))