CUDA_VISIBLE_DEVICES=0,1 python main_mxn_c.py --n 5 --m 3
#CUDA_VISIBLE_DEVICES=0,1 python main_mxn_c.py --n 7 --m 3
#CUDA_VISIBLE_DEVICES=0,1 python main_mxn_c.py --n 9 --m 3

CUDA_VISIBLE_DEVICES=0,1 python main_mxn_d.py --n 5 --m 3
#CUDA_VISIBLE_DEVICES=0,1 python main_mxn_d.py --n 7 --m 3
#CUDA_VISIBLE_DEVICES=0,1 python main_mxn_d.py --n 9 --m 3
#RuntimeError: CUDA error: invalid configuration argument
#CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
#For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
#Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
