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
