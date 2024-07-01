import os

zero_is_available = "SPACES_ZERO_GPU" in os.environ

if zero_is_available:
    import spaces  # type: ignore

    print("ZeroGPU is available")
else:
    print("ZeroGPU is not available")


# a decorator that applies the spaces.GPU decorator if zero is available
def zero(func):
    if zero_is_available:
        return spaces.GPU(func)
    else:
        return func
