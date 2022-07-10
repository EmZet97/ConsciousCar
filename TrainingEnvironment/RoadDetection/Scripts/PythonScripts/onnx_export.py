import torch

import torchvision
import onnx

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def load_model(src):
    model = torch.load(src)

    return model

def export_model(model, dest_src):
    input_names = ["input"]
    output_names = ["boxes", "labels", "scores", "masks"]

    dummy_input = torch.rand(1, 3, 512, 512).to(device)
    print(dummy_input)
    model.to(device)
    
    torch.onnx.export(model, 
                  dummy_input,
                  dest_src,
                  opset_version=12,
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=True,
                  )

    print(">>>>>>>>>> SAVED <<<<<<<<<<")
    model = onnx.load(dest_src)
    model.graph.output[3].type.tensor_type.shape.dim[0].dim_param = '?'
    onnx.save(model, dest_src)

def main():
    model = load_model("../Models/model.t")
    model.eval()

    export_model(model, "../Models/model.onnx")

main()