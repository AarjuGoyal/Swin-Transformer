import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from models.swin_transformer import SwinTransformer
import json
import urllib
import time
from collections import defaultdict
from PTQ_implementation import PTQ_implementor as PTQ
# from torch import 

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

class Profiler:
    
    def __init__(self, model, device ="cpu", run_with_hooks=False, apply_quantization=False):
        self.model = model
        self.device = device
        self.hooks = []
        self.layer_times = defaultdict(list)
        self.start_events = {}
        self.end_events = {}
        self.run_with_hooks = run_with_hooks
        if apply_quantization == True:
            self.PTQ = PTQ() 
        
        torch.cuda.empty_cache()
        if self.run_with_hooks:
            print("Run with hooks")
            self.register_cuda_hooks()

        self.keyword_list = ['.qkv', '.proj', '.fc1', '.fc2', 
                            '.norm', 'patch_embed.proj',
                            'reduction', 'head']

    def run_inference(self, image_tensor, iterations=1):
                 
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        self.model.eval()
        self.model= self.model.to(self.device)
        # Warmup
        for _ in range(10):
            _ = self.model(image_tensor)
        
        #Clear warm up times metric
        self.layer_times.clear()
        self.start_events.clear()
        self.end_events.clear()
        
        time.sleep(3)
        iter_time = []
        
        
        torch.cuda.synchronize()

        
        with torch.no_grad():
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = self.model(image_tensor)
                end.record()
                torch.cuda.synchronize()
                it = start.elapsed_time(end)
                iter_time.append(it)
            if self.run_with_hooks:
                self.compute_timings()
            # probabilities = torch.nn.functional.softmax(output[0], dim=0)
        

        elapsed_time = sum(iter_time)/iterations
        # elapsed_time = 0
        return elapsed_time

    def apply_quantization(self, type="dynamic", data_path=""):
        if type=="dynamic":
            self.model = PTQ.dynamic_quantization(model = self.model)
        return

    def register_cuda_hooks(self):

        def make_forward_pre_hook(name):
            """Hook that fire before forward layer execution"""
            def hook(module, input):
                if name not in self.start_events:
                    self.start_events[name] = torch.cuda.Event(enable_timing=True)
                self.start_events[name].record()
            return hook
        
        def make_forward_post_hook(name):
            """Hook the fires after forward layer execution"""
            def hook(module, input, output):
                if name not in self.end_events:
                    self.end_events[name] = torch.cuda.Event(enable_timing=True)
                self.end_events[name].record()
            return hook
        
        for name, module in self.model.named_modules():
            pre_hook = module.register_forward_pre_hook(make_forward_pre_hook(name))
            post_hook = module.register_forward_hook(make_forward_post_hook(name))
            self.hooks.append(pre_hook)
            self.hooks.append(post_hook)

    def extract_leaf_layer_results(self):
        leaf_layers = {}

        for (name, metrics) in self.layer_times.items():
            if any(keyword in name for keyword in self.keyword_list):
                leaf_layers[name] = sum(metrics)
        
        # Aggregate by type
        totals = {
        'Attention QKV': 0,
        'Attention Proj': 0,
        'MLP FC1': 0,
        'MLP FC2': 0,
        'LayerNorm': 0,
        'Patch Embed': 0,
        'Other': 0
        }

        total_layer_sum = 0.0
        for (name, time) in leaf_layers.items():
            print(f"{name:<50} {time}")
            if '.attn.qkv' in name:
                totals['Attention QKV'] += time
            elif '.attn.proj' in name:
                totals['Attention Proj'] += time
            elif '.mlp.fc1' in name:
                totals['MLP FC1'] += time
            elif '.mlp.fc2' in name:
                totals['MLP FC2'] += time
            elif '.norm' in name or 'norm' == name:
                totals['LayerNorm'] += time
            elif 'patch_embed' in name or 'reduction' in name:
                totals['Patch Embed'] += time
            else:
                totals['Other'] += time
        
        total = sum(totals.values())

        print("\n" + "="*70)
        print("AGGREGATED PROFILING (Leaf Layers Only)")
        print("="*70)
        print(f"{'Layer Type':<30} {'Time (ms)':<15} {'Percent':<10}")
        print("-"*70)

        for layer_type, time_ms in sorted(totals.items(), key=lambda x: x[1], reverse=True):
            if time_ms > 0:
                percent = (time_ms / total) * 100
                print(f"{layer_type:<30} {time_ms:>10.2f}      {percent:>8.1f}%")

        print("-"*70)
        print(f"{'TOTAL':<30} {total:>10.2f}      {'100.0':>8}%")

        # print("\n" + "="*70)
        # print("QUANTIZATION PRIORITY")
        # print("="*70)

        # mlp_total = totals['MLP FC1'] + totals['MLP FC2']
        # attn_total = totals['Attention QKV'] + totals['Attention Proj']

        # print(f"1. MLP layers (fc1 + fc2):        {mlp_total:>6.2f} ms ({mlp_total/total*100:>5.1f}%)")
        # print(f"2. Attention layers (qkv + proj): {attn_total:>6.2f} ms ({attn_total/total*100:>5.1f}%)")
        # print(f"3. Combined target for quantization: {mlp_total + attn_total:>6.2f} ms ({(mlp_total + attn_total)/total*100:>5.1f}%)")
        # print(f"Sum of inference time of all layers is {total_layer_sum}")
        

    def print_layer_profile(self, total_time):
        
        """Print detailed layer-by-layer breakdown"""
        print("\n" + "="*80)
        print("DETAILED LAYER-WISE PROFILING")
        print("="*80)
        print(f"Total inference time: {total_time:.2f} ms\n")
        
        print(f"{'Layer Name':<50} {'Time (ms)':<12}")
        print("-"*80)
        
        total_layer_sum = 0.0
        for (name, metrics) in self.layer_times.items():
            print(f"{name:<50} {metrics}")
            total_layer_sum += sum(metrics)
        
        print(f"Sum of inference time of all layers is {total_layer_sum}")
            # if len(summary) > 20:
            #     print(f"\n... and {len(summary) - 20} more layers")

    def compute_timings(self):
        torch.cuda.synchronize()
        for name in self.start_events.keys():
            if name in self.end_events:
                self.layer_times[name].append(self.start_events[name].elapsed_time(self.end_events[name]))


    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __exit__(self):
        self.remove_hooks()

def load_swin_model():
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
    checkpoint = torch.load('swin_tiny_patch4_window7_224.pth', map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    # model.eval()

    # Move to GPU

    # model = model.to(device)
    return model

def print_class_labels(probabilities):
    # Load ImageNet labels
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    imagenet_labels = json.loads(urllib.request.urlopen(url).read())

    print(f"\nPredictions for {img_path}:")
    print("-" * 50)
    for i in range(5):
        class_id = top5_catid[i].item()
        prob = top5_prob[i].item()
        print(f"{i+1}. {imagenet_labels[class_id]:25s} ({prob*100:.2f}%)")

def main():
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda'
    iterations = 10
    run_with_hooks = False
    apply_quantization = False

    img_path = 'data/test_image/dog.jpeg'
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    model = load_swin_model()
    Swin_Transformer_Profiler = Profiler(model=model, device=device, run_with_hooks=run_with_hooks, apply_quantization=apply_quantization)

    if apply_quantization == True:
        Swin_Transformer_Profiler.apply_quantization(type="dynamic")
    
    elapsed_time = Swin_Transformer_Profiler.run_inference(image_tensor=img_tensor, iterations=iterations)
    print(f"Time per iteratio is {elapsed_time}ms")
    if run_with_hooks == True:
        Swin_Transformer_Profiler.extract_leaf_layer_results()

    
if __name__== "__main__":
    main()