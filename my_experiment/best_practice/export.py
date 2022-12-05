import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ..deploy import load_model
import netron

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pth_file',
                        nargs='+',
                        type=str,
                        default='checkpoint.pth',
                        help='model path')
    parser.add_argument('-s',
                        '--source',
                        type=str,
                        default='data/images',
                        help='file/dir/0(webcam)')

    opt = parser.parse_args()

    for pth in opt.pth_file:
        model: nn.Module = load_model(pth)
        model.eval()

        # Input to the model
        x = torch.randn(1, 1, 224, 224, requires_grad=False)
        torch_out = model(x)
        print(torch_out)
        onnx_path = os.path.splitext(pth)[0] +".onnx"
        export_onnx(model)
        


def export_onnx(model, im, file, opset, dynamic, simplify, prefix='ONNX:'):
    import onnx

    print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
    f = file.with_suffix('.onnx')

    output_names = ['output0']
    # if dynamic:
    #     dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
    #     if isinstance(model, SegmentationModel):
    #         dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
    #         dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
    #     elif isinstance(model, DetectionModel):
    #         dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Metadata
    # d = {'stride': int(max(model.stride)), 'names': model.names}
    # for k, v in d.items():
    #     meta = model_onnx.metadata_props.add()
    #     meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

   
    return f, model_onnx

if __name__ == "__main__":
    main()
