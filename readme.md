# Torch ImageHash
**GPU Accelerate** algorithms from [imagehash](https://github.com/JohannesBuchner/imagehash)

## PHash
```
>>> # convert PIL img to tensor
>>> pic = PIL.Image.open("./Desktop/fat_bird.jpeg")
>>> img = torch.as_tensor(np.array(pic, copy=True))
>>> img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
>>> X = img.permute((2, 0, 1))
>>> 
>>> # forward prp
>>> model = PHasher()
>>> model_pil = PHasherPIL()
>>> 
>>> torch_result = model(X)
>>> torch_pil_result = model_pil(pic)
>>> phash_result = phash(pic)
>>> print("TorchScript vs PIL", 64 - (torch_result.numpy() == phash(pic)).sum())
TorchScript vs PIL 6
>>> print("TorchPIL vs PIL", 64 - (torch_pil_result.numpy() == phash(pic)).sum())
TorchPIL vs PIL 0
```

## Benchmark
```
running cpu benchmark: 100%|████████████████████████████████████| 1000/1000 [00:01<00:00, 647.11it/s]
CPU PHash: AVG RUNTIME:0.0015417692661285401

running gpu benchmark: 100%|███████████████████████████████████| 1000/1000 [00:00<00:00, 1485.72it/s]
GPU PHash: AVG RUNTIME:0.0006704351902008057

GPU was 2.2996544463406767 faster
```

**Versions**
```
Torch 1.10.0 (Anything with torch.fft.ftt)
Cuda 11.2 (Anything compatible w. Torch)
Pillow 6.2.1
GPU 2080 T.I
```
