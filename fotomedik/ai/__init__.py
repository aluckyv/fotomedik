"""Optional AI enhancement of repaired photos (a separate "_ai" copy).

Pretrained models, no training: Real-ESRGAN (super-resolution) sharpens the
areas that were rebuilt from the thumbnail, and LaMa (inpainting) fills holes
nothing is known about. Both run through ONNX Runtime: on the graphics card
with DirectML on Windows, on the processor otherwise. The detail they add is
invented, so the result is always a separate file next to the faithful repair.
"""
