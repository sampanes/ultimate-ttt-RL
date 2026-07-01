import random

NAMED_CONFIGS = {
    "small":  dict(conv_channels=[32, 64, 64],        fc_hidden_sizes=[256, 512, 256]),
    "medium": dict(conv_channels=[64, 128, 128, 256], fc_hidden_sizes=[512, 512, 256]),
    "large":  dict(conv_channels=[64, 128, 256, 256], fc_hidden_sizes=[512, 1024, 512, 256]),
}


def get_random_config(depth: int) -> dict:
    return {
        "conv_channels": [random.choice([32, 64, 128, 256]) for _ in range(depth)],
        "fc_hidden_sizes": [random.choice([128, 256, 512, 1024])
                            for _ in range(random.randint(2, 4))],
    }


def arch_label(conv_channels: list, fc_hidden_sizes: list) -> str:
    """Human-readable label. e.g. 'conv3/fc3-512'"""
    n_conv = len(conv_channels)
    n_fc = len(fc_hidden_sizes)
    max_fc = max(fc_hidden_sizes) if fc_hidden_sizes else 0
    return f"conv{n_conv}/fc{n_fc}-{max_fc}"
