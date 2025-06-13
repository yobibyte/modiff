import tabulate
import torch
from safetensors.torch import load_file

METRIC_NORM_DIFFERENCE = "norm_difference"
AVAILABLE_METRICS = {METRIC_NORM_DIFFERENCE}


def get_metrics(layer):
    return {
        METRIC_NORM_DIFFERENCE: torch.norm(layer).float().numpy(),
    }


def compare(
    chkpt1_path: str,
    chkpt2_path: str,
    filter_words: list[str] | None = None,
    sort_by: str | None = None,
):
    """Compare two checkpoints. Only single-file safetensors for now.

    The differences will be of the form second - first, e.g. if the first one is base, and the second is instruct, we will check how instruct is different wrt to the base.

    Args:
        chkpt1_path: Path to the first checkpoint.
        chkpt2_path: Path to the second checkpoint.
        filter_words: If a layer name contains one of these, it will not be skipped from the comparison.
        sort_by: Sort the table by this metric. Should be one of the AVAILABLE_METRICS.
    """
    c1 = load_file(chkpt1_path)
    c2 = load_file(chkpt2_path)
    assert not sort_by or sort_by in AVAILABLE_METRICS

    rows = []
    lns = list(c1.keys())
    if filter_words:
        lns = [el for el in lns if all(fw not in el for fw in filter_words)]

    metric_keys = []
    for ln in lns:
        ln1 = get_metrics(c1[ln])
        ln2 = get_metrics(c2[ln])
        metric_keys = list(ln1.keys())
        crow = [ln] + [ln2[k] - ln1[k] for k in metric_keys]
        rows.append(crow)

    if sort_by:
        # 1+ is because the first column is the layer name.
        rows = sorted(rows, key=lambda x: x[1 + metric_keys.index(sort_by)])

    return tabulate.tabulate(rows, headers=("layer",) + tuple(metric_keys))
