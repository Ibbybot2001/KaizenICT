
import torch
import sys

def check_cuda():
    print("Python Version:", sys.version)
    print("PyTorch Version:", torch.__version__)
    
    if torch.cuda.is_available():
        print(f"CUDA Available: YES")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
        # Simple tensor operation test
        x = torch.rand(5, 5).cuda()
        y = torch.rand(5, 5).cuda()
        z = x @ y
        print("Tensor Test (GPU): SUCCESS")
        print(z)
    else:
        print("CUDA Available: NO")
        print("This machine does not have a configured GPU for PyTorch.")

if __name__ == "__main__":
    check_cuda()
