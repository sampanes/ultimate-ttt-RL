import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

def load_state(path):
    raw = torch.load(path, map_location="cpu")
    return raw["model_state_dict"] if "model_state_dict" in raw else raw

def describe(state):
    print("\nLayer Statistics:")
    for name, param in state.items():
        data = param.float()
        print(f"{name:35}  shape={str(tuple(data.shape)):18}  mean={data.mean():.5f}  std={data.std():.5f}")

def diff_states(state1, state2):
    print("\nChanged Layers:")
    for name in state1:
        if name not in state2:
            print(f"  {name} only in first file")
            continue
        if state1[name].shape != state2[name].shape:
            print(f"  {name} shape mismatch: {state1[name].shape} vs {state2[name].shape}")
            continue
        diff = (state1[name] - state2[name]).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        if max_diff > 1e-6:
            print(f"  {name:35} max diff = {max_diff:.6f}, mean diff = {mean_diff:.6f}")
    for name in state2:
        if name not in state1:
            print(f"  {name} only in second file")

def state_diff_heatmap(state1, state2, title="Weight Changes", gamma=0.9):
    diffs, names, max_len = [], [], 0
    for key in state1:
        if key in state2 and state1[key].shape == state2[key].shape:
            delta = (state2[key] - state1[key]).flatten().numpy()
            names.append(key)
            diffs.append(delta)
            max_len = max(max_len, len(delta))

    padded = []
    for delta in diffs:
        pad = (max_len - len(delta)) // 2
        padded.append(np.pad(delta, (pad, max_len - len(delta) - pad), constant_values=0))

    matrix = np.stack(padded, axis=1)
    percentile = 99.5  # or 99
    vmax = np.percentile(np.abs(matrix), percentile)

    fig, ax = plt.subplots(figsize=(len(names) * 0.6 + 2, 8), facecolor="white")

    norm = plt.Normalize(-vmax, vmax)
    cmap = LinearSegmentedColormap.from_list("gray_centered", ["blue", "gray", "red"])

    # Apply gamma to exaggerate small differences, small gamma = exaggerate changes, big gamma = natural
    matrix_gamma = np.sign(matrix) * (np.abs(matrix) ** gamma)

    im = ax.imshow(matrix_gamma, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticks([])
    ax.set_title(title)

    for x in range(1, len(names)):
        ax.axvline(x - 0.5, color='gray', lw=0.3, linestyle='--', alpha=0.4)

    fig.colorbar(im, ax=ax, label="Weight Change")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1", required=True, help="Path to first .pt file")
    parser.add_argument("--f2", help="Optional second .pt file for comparison")
    parser.add_argument("--heatmap", action="store_true", help="Show heatmap diff if both files are provided")
    parser.add_argument("--gamma", help="If heatmap, choose gamma, .99 for accurate, .01 for exaggerated contrast")

    args = parser.parse_args()

    state1 = load_state(args.f1)
    describe(state1)

    if args.f2:
        state2 = load_state(args.f2)
        diff_states(state1, state2)

        if args.heatmap:
            title = f"Δ Weights\n{args.f1} vs {args.f2}"
            state_diff_heatmap(state1, state2, title, args.gamma)
