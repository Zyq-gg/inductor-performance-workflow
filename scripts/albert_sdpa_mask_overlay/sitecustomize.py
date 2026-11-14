"""Opt-in diagnostic fix for mask-free bidirectional Transformers SDPA.

Enable with PYTHONPATH=<directory containing this file> for the benchmark process.
Tested with Transformers 5.10.1; this is not an upstream package patch.
"""

from transformers import masking_utils


_original_ignore_bidirectional_mask_sdpa = masking_utils._ignore_bidirectional_mask_sdpa


def _ignore_bidirectional_mask_sdpa_with_static_none(
    padding_mask, kv_length, local_attention_size=None
):
    # This case is all-visible even while Dynamo traces the model.
    if padding_mask is None and local_attention_size is None:
        return True
    return _original_ignore_bidirectional_mask_sdpa(
        padding_mask, kv_length, local_attention_size
    )


masking_utils._ignore_bidirectional_mask_sdpa = (
    _ignore_bidirectional_mask_sdpa_with_static_none
)
