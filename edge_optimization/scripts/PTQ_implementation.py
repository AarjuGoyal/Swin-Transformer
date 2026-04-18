import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'original'))

script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(script_dir, '..')
import torch
import torch.nn as nn
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_tensorrt_llm_checkpoint
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from models.swin_transformer import SwinTransformer

def load_model(device):
    
    model = SwinTransformer(
    img_size=224, patch_size=4, in_chans=3, num_classes=1000,
    embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24], window_size=7
    )
    checkpoint = torch.load(os.path.join(BASE_DIR, 'models', 'swin_tiny_patch4_window7_224.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval().to(device)
    return model

def calibration_Dataloader():
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    calib_dataset = datasets.ImageFolder(os.path.join(BASE_DIR, 'data', 'imagenet_val'), transform=transform)
    calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=True, num_workers=4)
    return calib_loader

def calibrate_fn(model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calib_loader = calibration_Dataloader()
    model.eval()
    for i, (images, _) in enumerate(calib_loader):
        model(images.to(device))
        if i >= 15:
            break

def main():
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(device)
    
    # calibrate_fn(model, calib_loader)

    
    #Quantization
    model = mtq.quantize(model, mtq.INT8_SMOOTHQUANT_CFG, forward_loop=calibrate_fn)
    mtq.print_quant_summary(model)

    print("Exporting quantized ONNX...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        'swin_int8_smoothquant.onnx',
        input_names=['input'],
        output_names=['output'],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Done — swin_int8_smoothquant.onnx saved")

if __name__ == "__main__":
    main()
# model = mtq.quantization("transormer_models/Swin-Transformer/edge_optimization/swin_tiny_fp32.onnx")

'''
class PTQ_implementor:
    def get_calibration_dataloader(self, data_path, batch_size=16, num_samples=100):
        """
        Create calibration dataset
        """
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
        # Load dataset (use ImageNet validation or Tiny-ImageNet)
        # Replace with your dataset path
        dataset = datasets.ImageFolder(data_path, transform=transform)
        
        # Use only a subset for calibration
        subset = Subset(dataset, range(min(num_samples, len(dataset))))
        
        loader = DataLoader(subset, batch_size=batch_size, 
                        shuffle=False, num_workers=4)
        
        return loader

    def dynamic_quantization( model):
        """
        Quantization Linear Layers only
        - Quantize weights statically,
        - Quantize activations dynamically
        - No calibration used
        """
        model = model.to("cpu")
        torch.backends.quantized.engine = 'qnnpack'
        model_quantized = quant.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype = torch.qint8
        )

        return model_quantized
'''
    