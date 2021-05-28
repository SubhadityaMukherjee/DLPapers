- Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., ... & Wu, H. (2017). Mixed precision training. arXiv preprint arXiv:1710.03740.
[Paper](https://arxiv.org/pdf/1710.03740)

# When do we face an issue
Most deep learning frameworks, including PyTorch, train with 32-bit floating point (FP32) arithmetic by default. However this is not essential to achieve full accuracy for many deep learning models. 

NVIDIA = achieved the same accuracy as FP32 training using the same hyperparameters, with additional performance benefits on NVIDIA GPUs:

Shorter training time;
Lower memory requirements, enabling larger batch sizes, larger models, or larger inputs.





# Why do we really care



# What are we doing



# Simple implementation 
- Pytorch, lightning

Pytorch
```python
import torch
# Creates once at the beginning of training
scaler = torch.cuda.amp.GradScaler()

for data, label in data_iter:
   optimizer.zero_grad()
   # Casts operations to mixed precision
   with torch.cuda.amp.autocast():
      loss = model(data)

   # Scales the loss, and calls backward()
   # to create scaled gradients
   scaler.scale(loss).backward()

   # Unscales gradients and calls
   # or skips optimizer.step()
   scaler.step(optimizer)

   # Updates the scale for next iteration
   scaler.update()
```

Lightning
```python
trainer = pl.Trainer(auto_select_gpus=True,
                     gpus=1,
                     precision=16,
                     profiler=False,
                     max_epochs=5,
                     callbacks=[pl.callbacks.ProgressBar()],
                     automatic_optimization=True,
                     enable_pl_optimizer=True,
                     logger=logger,
                     accelerator='ddp',
                     plugins='ddp_sharded')
`c ``
