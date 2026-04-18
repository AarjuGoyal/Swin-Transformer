import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',"original"))
import torch
import torch.nn as nn
import time
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models.swin_transformer import SwinTransformer
# from PTQ_implementation import PTQ_implementor as PTQ
image_dir = "data/imagenet_val/"

# os.listdir(image_dir)
# for d in os.listdir(image_dir):
#     dir_path = os.path.join(image_dir, d)
#     if os.path.isdir(dir_path):  # Check if the item is a directory
#         print(d, len(os.listdir(dir_path)))
        
#         print(os.listdir(dir_path)[:5])

# with open('data/words.txt') as f:
#     lines = f.readlines()
#     class2label = {l[:9].strip(): l[10:-1].strip() for l in lines}
#     class2ix = {l[:9].strip(): ix for ix, l in enumerate(lines)}
#     print(class2label)


class AccuracyTester:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def test_accuracy(self, val_loader, max_batches=None):
        """
        Test top-1 and top-5 accuracy
        """
        correct_top1 = 0
        correct_top5 = 0

        inference_times = []

        self.model = self.model.to(self.device)
        total_labels = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(val_loader)):
                if max_batches and batch_idx >= max_batches:
                    break
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                print(f"{images.shape} {labels.shape}")
                # Add this before running inference
                
                print(f"label: {labels}, img shape: {images.shape}, min: {images.min():.2f}, max: {images.max():.2f}")
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                outputs = self.model(images)
                end.record()

                torch.cuda.synchronize()
                inference_times.append(start.elapsed_time(end))

                _, pred_top1 = outputs.max(1)
                correct_top1 += pred_top1.eq(labels).sum().item()

                # print(f"{images.shape} {labels.shape} {outputs.shape} {pred_top1}")
                # Top-5 accuracy
                _, pred_top5 = outputs.topk(5, 1, largest=True, sorted=True)
                correct_top5 += pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5)).sum().item()
                
                total_labels += labels.size(0)

            top1_acc = 100. * correct_top1 / total_labels
            top5_acc = 100. * correct_top5 / total_labels
            avg_time = sum(inference_times) / len(inference_times)

            return {
                'top1_accuracy': top1_acc,
                'top5_accuracy': top5_acc,
                'total_samples': total_labels,
                'avg_inference_time': avg_time,
                'throughput_fps': 1000.0 / avg_time * images.size(0)  # Account for batch size
            }

def load_imagenet_val(data_path, batch_size=32, num_workers =4, device='cpu'):
    """
    Load ImageNet validation set
    """
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(data_path, transform=img_transforms)

    print(f"Loaded {len(val_dataset)} validation images")
    print(f"Number of classes: {len(val_dataset.classes)}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=True
    )

    return val_loader

def load_model(model_name, checkpoint_path, device="cpu"):
    if model_name == "SwinTransformer":
        model = SwinTransformer(
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=96,           # Tiny model
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    return model
    
def test_tensorrt_accuracy(engine_path, val_loader, max_batches=None):
    """
    Test TensorRT model accuracy
    Requires pycuda and tensorrt Python bindings
    """
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        import tensorrt as trt
        import numpy as np
    except ImportError:
        print("Error: pycuda or tensorrt not installed")
        print("Install with: pip install pycuda")
        return None
    
    # Load TensorRT engine
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    # context.set_input_shape('input', (val_loader.batch_size, 3, 224, 224))
    # Allocate buffers
    input_shape = (val_loader.batch_size, 3, 224, 224)
    output_shape = (val_loader.batch_size, 1000)
    
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    
    h_input = cuda.pagelocked_empty(1 * 3 * 224 * 224, dtype=np.float32)
    h_output = cuda.pagelocked_empty(1 * 1000, dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    print(f"\nTesting TensorRT accuracy on {len(val_loader)} batches...")
    
    for batch_idx, (images, labels) in enumerate(tqdm(val_loader)):


        if max_batches and batch_idx >= max_batches:
            break
        
        # Prepare input
        actual_batch_size = images.size(0)
        
        output_shape = (actual_batch_size, 1000)
        # if actual_batch_size != val_loader.batch_size:
        #     # Skip incomplete batch
        #     continue
        
        for i in range(images.size(0)):
            inp = images[i:i+1].cpu().contiguous().numpy().ravel().astype(np.float32)
        
            h_input = cuda.pagelocked_empty(1 * 3 * 224 * 224, dtype=np.float32)
            h_output = cuda.pagelocked_empty(1 * 1000, dtype=np.float32)
            d_input = cuda.mem_alloc(h_input.nbytes)
            d_output = cuda.mem_alloc(h_output.nbytes)
            
            context.set_input_shape('input', (1, 3, 224, 224))
            np.copyto(h_input, inp)
            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.set_tensor_address('input', int(d_input))
            context.set_tensor_address('output', int(d_output))
            context.execute_async_v3(stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
            
            out_tensor = torch.from_numpy(h_output.reshape(1, 1000))
            
            pred_top1 = out_tensor.max(1)[1]
            correct_top1 += pred_top1.eq(labels[i]).sum().item()
            
            pred_top5 = out_tensor.topk(5, 1)[1]
            correct_top5 += pred_top5.eq(labels[i].view(-1,1).expand_as(pred_top5)).sum().item()
            
            total += 1

    print(f'Top-1: {100.*correct_top1/total:.2f}%, Top-5: {100.*correct_top5/total:.2f}% on {total} samples')
    
    top1_acc = 100. * correct_top1 / total
    top5_acc = 100. * correct_top5 / total
    
    return {
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'total_samples': total
    }

def test_onnx_accuracy(onnx_path, val_loader):

    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(onnx_path)

    correct_top1 = 0
    total = 0
    correct = 0
    for batch_idx, (images, labels) in enumerate(tqdm(val_loader)):
        print(f"Value of batch_idx {batch_idx}, images shape {images.shape}, labels {labels}")
        print(f"images dtype: {images.dtype}")
        print(f"images min: {images.min():.3f}, max: {images.max():.3f}")
        print(f"images mean: {images.mean():.3f}, std: {images.std():.3f}")
        
        for i in range(images.size(0)):
            inp = images[i:i+1].cpu().contiguous().numpy().astype(np.float32)
            out = sess.run(None, {'input': inp})[0]
            pred = out.argmax()
            if pred == labels[i].item():
                correct += 1
            total += 1
        if batch_idx>=10:
            break
    print(f'Single image accuracy: {100.*correct/total:.2f}% on {total} samples')

        # print(f"inp shape: {inp.shape}, mean: {inp.mean():.3f}, std: {inp.std():.3f}")
        # print(f"labels: {labels[:5].tolist()}")
        # out = sess.run(None, {'input': inp})[0]
        # pred = torch.from_numpy(out).max(1)[1]
        # correct_top1 += pred.eq(labels).sum().item()
        # total += labels.size(0)
        # print(f"Prediction is {pred}")
    # print(f"Total correction prediction are {correct_top1}")
    # print(f'ONNX Top-1: {100. * correct_top1 / total:.2f}% on {total} samples')

def main():
    """Main testing function"""
    
    # Configuration
    IMAGENET_VAL_PATH = 'data/imagenet_val/'
    CHECKPOINT_PATH = 'models/swin_tiny_patch4_window7_224.pth'
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    MAX_BATCHES = None  # Set to e.g., 50 for quick testing (1600 images)
    MODEL_NAME = "SwinTransformer"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print("ACCURACY TESTING")
    print("="*70)

    # Load validation data
    print(f"\nLoading ImageNet validation set from: {IMAGENET_VAL_PATH}")
    ImageNet_Val_Loader = load_imagenet_val(IMAGENET_VAL_PATH, BATCH_SIZE, NUM_WORKERS, DEVICE)
    
    if MAX_BATCHES:
        print(f"Testing on {MAX_BATCHES * BATCH_SIZE} images (quick test)")
    else:
        print(f"Testing on full validation set ({len(ImageNet_Val_Loader.dataset)} images)")
    
    results = {}
    
    # Test PyTorch FP32
    print("\n" + "="*70)
    print("Testing PyTorch FP32 Model")
    print("="*70)
    # swin_model_fp32 = load_model("SwinTransformer", CHECKPOINT_PATH, DEVICE)
    # Accuracy_Tester_FP32 = AccuracyTester(swin_model_fp32, DEVICE)
    # results['pytorch_fp32'] = Accuracy_Tester_FP32.test_accuracy(ImageNet_Val_Loader, MAX_BATCHES)
    
    # # Test PyTorch INT8 (CPU only)
    # print("\n" + "="*70)
    # print("Testing PyTorch INT8 Model (CPU)")
    # print("="*70)
    # model_int8 = load_quantized_model(CHECKPOINT_PATH, 'cpu')
    
    # # Need to reload data for CPU
    # val_loader_cpu = load_imagenet_val(IMAGENET_VAL_PATH, BATCH_SIZE, NUM_WORKERS)
    # tester_int8 = AccuracyTester(model_int8, 'cpu')
    # results['pytorch_int8'] = tester_int8.test_accuracy(val_loader_cpu, MAX_BATCHES)

    # # print("\n" + "="*70)
    # # print("Testing TensorRT FP32 Model")
    # # print("="*70)
    # # results['tensorrt_fp32'] = test_tensorrt_accuracy(
    # #     'swin_fp32.trt', val_loader, MAX_BATCHES
    # # )
    
    print("\n" + "="*70)
    print("Testing TensorRT INT8 Model")
    print("="*70)
    print("Verify onnx results")
    # results['onnx_int8_smoothquant'] = test_onnx_accuracy(
    #     'swin_int8_smoothquant.onnx', ImageNet_Val_Loader
    # )
    results['tensorrt_int8'] = test_tensorrt_accuracy(
        'swin_int8_smoothquant.trt', ImageNet_Val_Loader, MAX_BATCHES
    )
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'Model':<25} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Samples':<10}")
    print("-"*70)
    
    for model_name, result in results.items():
        if result:
            print(f"{model_name:<25} {result['top1_accuracy']:>10.2f}%  "
                  f"{result['top5_accuracy']:>10.2f}%  {result['total_samples']:>8}")
    
    # # Calculate accuracy drop
    # if 'pytorch_fp32' in results and 'pytorch_int8' in results:
    #     fp32_top1 = results['pytorch_fp32']['top1_accuracy']
    #     int8_top1 = results['pytorch_int8']['top1_accuracy']
    #     drop = fp32_top1 - int8_top1
    #     print(f"\nAccuracy drop (FP32 → INT8): {drop:.2f}%")
    
    # # Save results to JSON
    # with open('accuracy_results.json', 'w') as f:
    #     json.dump(results, f, indent=2)
    
    # print("\n✓ Results saved to accuracy_results.json")


if __name__ == "__main__":
    main()

