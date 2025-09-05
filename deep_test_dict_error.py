import torch
import torch.nn as nn

class TestModule(nn.Module):
    def forward(self, x, return_all=False):
        outputs = {
            'head': torch.randn(1, 7),
            'advanced': torch.randn(1, 7)
        }
        
        if return_all:
            print(f"[Module] Returning dict: {type(outputs)}")
            return outputs
        else:
            return outputs['head']

# 테스트 1: 일반 호출
model = TestModule()
model.eval()

try:
    print("=== Test 1: Normal call ===")
    result = model(torch.randn(1, 10), return_all=True)
    print(f"Result type: {type(result)}")
    print(f"Success!")
except Exception as e:
    print(f"Error: {e}")

# 테스트 2: CUDA 이동 후 (GPU가 없으면 CPU 사용)
print("\n=== Test 2: After .to() ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

try:
    inputs = torch.randn(1, 10).to(device)
    result = model(inputs, return_all=True)
    print(f"Result type: {type(result)}")
    
    # 혹시 결과에 .to()를 자동으로 적용하려고 하나?
    if hasattr(result, 'to'):
        print("Result has .to() method")
    else:
        print("Result does NOT have .to() method")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
