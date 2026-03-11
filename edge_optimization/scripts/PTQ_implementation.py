import torch
import torch.nn as nn
import torch.quantization as quant
from torchvision import datasets, transforms
import time
import numpy as np

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

    