import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

from train import HybridForensicModel


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT = "hybrid_encoder_epoch_150.pth"

EVAL_ROOT = "/home/sm-signal-learning/24b1257/evaluation_set"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# HELPERS
# ============================================================

def extract_epoch_name(checkpoint_name):
    match = re.search(r"epoch_(\d+)", checkpoint_name)

    if match:
        return match.group(1)

    return "latest"


def load_image(path):
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, [512, 512])
    img = TF.to_tensor(img)

    return img.unsqueeze(0)


@torch.inference_mode()
def compute_embeddings(folder, model):

    paths = sorted(Path(folder).glob("*.jpg"))

    embeddings = []

    for idx, path in enumerate(paths):

        if idx % 20 == 0:
            print(f"Processing {folder}: {idx}/{len(paths)}")

        img = load_image(path).to(DEVICE)

        emb = model(img)

        emb = F.normalize(emb, dim=-1)

        embeddings.append(
            emb.squeeze(0).cpu()
        )

    return torch.stack(embeddings)


def compute_matrix(A, B):
    return (A @ B.T).numpy()


def compute_stats(M):

    diag = np.diag(M)

    offdiag_mask = ~np.eye(M.shape[0], dtype=bool)

    offdiag = M[offdiag_mask]

    stats = {

        # -------------------------
        # DIAGONAL
        # -------------------------

        "diag_mean": float(diag.mean()),
        "diag_std": float(diag.std()),
        "diag_max": float(diag.max()),
        "diag_min": float(diag.min()),

        # -------------------------
        # OFF-DIAGONAL
        # -------------------------

        "offdiag_mean": float(offdiag.mean()),
        "offdiag_std": float(offdiag.std()),
        "offdiag_max": float(offdiag.max()),
        "offdiag_min": float(offdiag.min()),

        # -------------------------
        # GAP
        # -------------------------

        "gap": float(diag.mean() - offdiag.mean())
    }

    return stats


def save_heatmap(M, title, path):

    plt.figure(figsize=(8, 7))

    plt.imshow(
        M,
        aspect="auto",
        interpolation="nearest"
    )

    plt.colorbar()

    plt.title(title)

    plt.xlabel("X")

    plt.ylabel("Y")

    plt.tight_layout()

    plt.savefig(path, dpi=300)

    plt.close()


def write_stats(fp, name, M):

    stats = compute_stats(M)

    diag = np.diag(M)

    fp.write("\n")
    fp.write("=" * 70 + "\n")
    fp.write(name + "\n")
    fp.write("=" * 70 + "\n")

    # =====================================================
    # DIAGONAL STATS
    # =====================================================

    fp.write("\n")
    fp.write("MATCHED PAIRS (DIAGONAL)\n")
    fp.write("-" * 40 + "\n")

    fp.write(f"Mean : {stats['diag_mean']:.6f}\n")
    fp.write(f"Std  : {stats['diag_std']:.6f}\n")
    fp.write(f"Max  : {stats['diag_max']:.6f}\n")
    fp.write(f"Min  : {stats['diag_min']:.6f}\n")

    # =====================================================
    # RANDOM DIAGONAL SAMPLES
    # =====================================================

    fp.write("\n")
    fp.write("5 RANDOM MATCHED PAIRS\n")

    random_diag_indices = random.sample(
        range(len(diag)),
        5
    )

    for idx in random_diag_indices:

        fp.write(
            f"({idx:03d},{idx:03d}) "
            f"-> {diag[idx]:.6f}\n"
        )

    # =====================================================
    # OFF-DIAGONAL STATS
    # =====================================================

    fp.write("\n")
    fp.write("UNMATCHED PAIRS (OFF-DIAGONAL)\n")
    fp.write("-" * 40 + "\n")

    fp.write(f"Mean : {stats['offdiag_mean']:.6f}\n")
    fp.write(f"Std  : {stats['offdiag_std']:.6f}\n")
    fp.write(f"Max  : {stats['offdiag_max']:.6f}\n")
    fp.write(f"Min  : {stats['offdiag_min']:.6f}\n")

    # =====================================================
    # RANDOM OFF-DIAGONAL SAMPLES
    # =====================================================

    fp.write("\n")
    fp.write("5 RANDOM UNMATCHED PAIRS\n")

    n = M.shape[0]

    count = 0

    while count < 5:

        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)

        if i == j:
            continue

        fp.write(
            f"({i:03d},{j:03d}) "
            f"-> {M[i,j]:.6f}\n"
        )

        count += 1

    # =====================================================
    # GAP
    # =====================================================

    fp.write("\n")
    fp.write(f"Gap (diag - offdiag): {stats['gap']:.6f}\n")


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = HybridForensicModel().to(DEVICE)

ckpt = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(ckpt["model"])

model.eval()

print("Checkpoint loaded.")

# ============================================================
# OUTPUT DIR
# ============================================================

epoch_name = extract_epoch_name(CHECKPOINT)

outdir = Path(
    f"results_epoch_{epoch_name}"
)

outdir.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# EMBEDDINGS
# ============================================================

real_emb = compute_embeddings(
    Path(EVAL_ROOT) / "real",
    model
)

positive_emb = compute_embeddings(
    Path(EVAL_ROOT) / "positive",
    model
)

negative_emb = compute_embeddings(
    Path(EVAL_ROOT) / "negative",
    model
)

torch.save(
    real_emb,
    outdir / "embeddings_real.pt"
)

torch.save(
    positive_emb,
    outdir / "embeddings_positive.pt"
)

torch.save(
    negative_emb,
    outdir / "embeddings_negative.pt"
)

# ============================================================
# MATRICES
# ============================================================

M_rp = compute_matrix(
    real_emb,
    positive_emb
)

M_rn = compute_matrix(
    real_emb,
    negative_emb
)

M_pn = compute_matrix(
    positive_emb,
    negative_emb
)

diag_rp = np.diag(M_rp)
diag_rn = np.diag(M_rn)

labels = np.concatenate([
    np.ones(len(diag_rp)),
    np.zeros(len(diag_rn))
])

scores = np.concatenate([
    diag_rp,
    diag_rn
])

thresholds = np.linspace(-1, 1, 2001)

best_acc = 0
best_threshold = 0

for t in thresholds:

    pred = (scores >= t).astype(int)

    acc = (pred == labels).mean()

    if acc > best_acc:
        best_acc = acc
        best_threshold = t

best_rn_idx = np.argsort(diag_rn)[-10:]
worst_rn_idx = np.argsort(diag_rn)[:10]

np.savez(
    outdir / "similarity_matrices.npz",
    real_positive=M_rp,
    real_negative=M_rn,
    positive_negative=M_pn
)

# ============================================================
# HEATMAPS
# ============================================================

save_heatmap(
    M_rp,
    "Real vs Positive",
    outdir / f"heatmap_real_positive_epoch_{epoch_name}.png"
)

save_heatmap(
    M_rn,
    "Real vs Negative",
    outdir / f"heatmap_real_negative_epoch_{epoch_name}.png"
)

save_heatmap(
    M_pn,
    "Positive vs Negative",
    outdir / f"heatmap_positive_negative_epoch_{epoch_name}.png"
)

# ============================================================
# STATS
# ============================================================

with open(
    outdir / "statistics.txt",
    "w"
) as fp:

    fp.write(
        f"Checkpoint Epoch: {epoch_name}\n"
    )

    write_stats(
        fp,
        "REAL vs POSITIVE",
        M_rp
    )

    write_stats(
        fp,
        "REAL vs NEGATIVE",
        M_rn
    )

    write_stats(
        fp,
        "POSITIVE vs NEGATIVE",
        M_pn
    )

    fp.write("\n")
    fp.write("=" * 70 + "\n")
    fp.write("REAL-NEGATIVE HARDEST / EASIEST CASES\n")
    fp.write("=" * 70 + "\n")

    fp.write("\nTOP 10 RN SIMILARITIES\n")

    for idx in reversed(best_rn_idx):

        fp.write(
            f"{idx:03d} -> {diag_rn[idx]:.6f}\n"
        )

    fp.write("\nBOTTOM 10 RN SIMILARITIES\n")

    for idx in worst_rn_idx:

        fp.write(
            f"{idx:03d} -> {diag_rn[idx]:.6f}\n"
        )

    fp.write("\n")
    fp.write("=" * 70 + "\n")
    fp.write("THRESHOLD ANALYSIS\n")
    fp.write("=" * 70 + "\n")

    fp.write(
        f"Best Threshold : {best_threshold:.6f}\n"
    )

    fp.write(
        f"Best Accuracy  : {100*best_acc:.2f}%\n"
    )

print()
print("=" * 70)
print("Evaluation complete")
print(f"Results saved to: {outdir}")
print("=" * 70)