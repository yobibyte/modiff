from modiff import compare
from modiff.metrics import METRIC_NORM_DIFFERENCE

if __name__ == "__main__":
    chkpt1_path = "/home/yobibyte/Downloads/qwen2p5_0p5b_base.safetensors"
    chkpt2_path = "/home/yobibyte/Downloads/qwen2p5_0p5b_instruct.safetensors"
    stats = compare(
        chkpt1_path, chkpt2_path, filter_words=["bias"], sort_by=METRIC_NORM_DIFFERENCE
    )
    print(stats)
