import torch
import torch.onnx
import onnx
import onnxruntime as ort
from models.swin_transformer import SwinTransformer
def onnx_export(model, dummy_input, output_file):
    torch.onnx.export(
        model,
        dummy_input,
        f = output_file,
        input_names = ['input'],
        output_names = ['output'],
        opset_version = 17,
        export_params = True,
        dynamic_axes = {                      # Dynamic batch size
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        do_constant_folding = True,
    )
    # torch.onnx.export
    # (
    #     model,                              # Model
    #     dummy_input,                        # Example input
    #     output_file,
    #     export_params=True,
    #     opset_version=17,

    #     opset_version=17,
    #     do_constant_folding=True,           # Optimize constants
    #     input_names=['input'],              # Input names
    #     output_names=['output'],            # Output names
    #     dynamic_axes=
    #     {                      # Dynamic batch size
    #         'input': {0: 'batch_size'},
    #         'output': {0: 'batch_size'}
    #     }
    # )

def verify_onnx_model():
    return

def main():
    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SwinTransformer(
        img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=7
    )
    checkpoint = torch.load('swin_tiny_patch4_window7_224.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # 3. Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✓ Output shape: {output.shape}")

    onnx_export(model, dummy_input, "swin_tiny_fp32.onnx")
    
if __name__ == "__main__":
    main()

