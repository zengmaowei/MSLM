#!/usr/bin/env python3
"""
MSSM FLOPs 计算脚本 - 改进版
支持多种输入尺寸和详细的层分析
"""

import torch
import torch.nn as nn
import time
import numpy as np
from ptflops import get_model_complexity_info
from thop import profile
from basicsr.models.archs.MSSM import MSSM, count_param

def calculate_flops_detailed(input_sizes=[(3, 256, 256), (3, 512, 512), (3, 1024, 1024)]):
    """详细计算不同输入尺寸下的 FLOPs"""
    
    print("="*80)
    print("MSSM Model FLOPs Analysis")
    print("="*80)
    
    # 创建模型
    model = MSSM().cuda()
    model.eval()
    
    # 基本参数信息
    total_params = count_param(model)
    print(f"Total Parameters: {total_params/1e6:.2f} M")
    print(f"Model Size: {total_params * 4 / 1024**2:.2f} MB (float32)")
    print()
    
    # 不同输入尺寸的 FLOPs
    for i, input_size in enumerate(input_sizes):
        print(f"Input Size: {input_size}")
        print("-" * 40)
        
        # 创建随机输入
        input_tensor = torch.randn(1, *input_size).cuda()
        
        # 方法1: ptflops
        try:
            macs_ptflops, params_ptflops = get_model_complexity_info(
                model,
                input_size,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False
            )
            flops_ptflops = macs_ptflops * 2
            
            print(f"ptflops - MACs: {macs_ptflops/1e9:.2f} G, FLOPs: {flops_ptflops/1e9:.2f} G")
        except Exception as e:
            print(f"ptflops failed: {e}")
            flops_ptflops = None
        
        # 方法2: thop
        try:
            macs_thop, params_thop = profile(model, inputs=(input_tensor,), verbose=False)
            flops_thop = macs_thop * 2
            
            print(f"thop    - MACs: {macs_thop/1e9:.2f} G, FLOPs: {flops_thop/1e9:.2f} G")
        except Exception as e:
            print(f"thop failed: {e}")
            flops_thop = None
        
        # 推理时间测试
        try:
            num_runs = 20
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(input_tensor)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            fps = 1.0 / avg_time
            
            print(f"Inference - Time: {avg_time*1000:.2f} ms, FPS: {fps:.2f}")
            
            # 计算实际 FLOPs (基于时间和理论峰值)
            # 假设 GPU 理论峰值约为 20 TFLOPs (RTX 4090)
            theoretical_peak_flops = 20e12  # 20 TFLOPs
            actual_flops = flops_ptflops if flops_ptflops is not None else flops_thop
            if actual_flops is not None:
                efficiency = (actual_flops / theoretical_peak_flops) * 100
                print(f"Hardware Efficiency: {efficiency:.2f}%")
                
        except Exception as e:
            print(f"Inference test failed: {e}")
        
        print()
    
    # 输出尺寸分析
    print("Output Analysis:")
    print("-" * 40)
    test_input = torch.randn(1, 3, 256, 256).cuda()
    
    with torch.no_grad():
        outputs = model(test_input)
    
    for i, out in enumerate(outputs):
        print(f"Output {i+1}: {out.shape}")
    
    print()
    
    # 内存使用分析
    print("Memory Usage:")
    print("-" * 40)
    
    # 测试不同 batch size 的内存使用
    for batch_size in [1, 2, 4, 8]:
        try:
            test_input = torch.randn(batch_size, 3, 256, 256).cuda()
            
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(test_input)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            print(f"Batch {batch_size}: Peak Memory: {peak_memory:.2f} GB")
            
        except Exception as e:
            print(f"Batch {batch_size}: Failed - {e}")
    
    print("="*80)

def analyze_model_components():
    """分析模型各组件的参数量"""
    
    model = MSSM()
    
    print("Component Analysis:")
    print("-" * 40)
    
    # 主要组件分析
    components = {
        'multiscale': model.multiscale,
        'patch_embed': model.patch_embed,
        'output': model.output,
    }
    
    for name, component in components.items():
        params = sum(p.numel() for p in component.parameters())
        print(f"{name:15s}: {params/1e6:.2f} M params ({params/total_params*100:.1f}%)")
    
    print()

if __name__ == "__main__":
    calculate_flops_detailed()
    analyze_model_components()
