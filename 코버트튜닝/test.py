import torch

# GPU 사용 가능 여부 확인
print("CUDA Available:", torch.cuda.is_available())

# GPU 개수 확인
print("Available GPUs:", torch.cuda.device_count())

# 첫 번째 GPU 이름 출력
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
else:
    print("No GPU detected.")
