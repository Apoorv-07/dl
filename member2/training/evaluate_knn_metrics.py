import os
import sys
import torch
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)

from models.member2_model import Member2FoundationModel


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "graph_validation.pt"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints_validation",
    "best_model.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

K = 5
SEED = 42


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
print("=" * 70)
print("PROPER KNN ANOMALY DETECTION EVALUATION")
print("=" * 70)

print(f"\nDevice: {DEVICE}")
print(f"k: {K}")


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------
data = torch.load(
    DATASET_PATH,
    map_location="cpu",
    weights_only=False
)

sequences = data["sequences"]

print(f"Total sequences: {len(sequences)}")


# ---------------------------------------------------------
# LOAD CHECKPOINT
# ---------------------------------------------------------
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Best epoch: {checkpoint['best_epoch']}")
print(
    f"Best validation loss: "
    f"{checkpoint['best_validation_loss']}"
)


# ---------------------------------------------------------
# CREATE MODEL
# ---------------------------------------------------------
model = Member2FoundationModel(
    in_channels=53,
    hidden_channels=128,
    heads=4,
    transformer_heads=4,
    transformer_layers=2,
    h_dim=128,
    z_dim=64,
    dropout=0.2
)


# ---------------------------------------------------------
# LOAD GAT
# ---------------------------------------------------------
gat_state_dict = checkpoint["gat_state_dict"]

gat1_state_dict = {
    key.replace("gat1.", "", 1): value
    for key, value in gat_state_dict.items()
    if key.startswith("gat1.")
}

gat2_state_dict = {
    key.replace("gat2.", "", 1): value
    for key, value in gat_state_dict.items()
    if key.startswith("gat2.")
}

model.gat1.load_state_dict(gat1_state_dict)
model.gat2.load_state_dict(gat2_state_dict)

model.temporal_transformer.load_state_dict(
    checkpoint["transformer_state_dict"]
)

model.projection_head.load_state_dict(
    checkpoint["projection_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("\nModel loaded successfully.")


# ---------------------------------------------------------
# GENERATE H EMBEDDINGS
# ---------------------------------------------------------
print("\nGenerating h embeddings...")

embeddings = []
labels = []

with torch.no_grad():

    for i, sequence in enumerate(sequences):

        snapshots = [
            snapshot.to(DEVICE)
            for snapshot in sequence["snapshots"]
        ]

        output = model.encode_sequence(snapshots)

        h = output["h"]

        embeddings.append(
            h.squeeze(0).cpu()
        )

        labels.append(
            int(sequence["label"])
        )

        if (i + 1) % 500 == 0:
            print(
                f"Processed {i + 1}/{len(sequences)}"
            )


embeddings = torch.stack(embeddings).numpy()
labels = np.asarray(labels)

print(
    f"\nEmbedding shape: {embeddings.shape}"
)


# ---------------------------------------------------------
# BENIGN / ATTACK COUNTS
# ---------------------------------------------------------
benign_indices = np.where(labels == 0)[0]
attack_indices = np.where(labels == 1)[0]

print("\nDataset distribution:")
print(f"Benign: {len(benign_indices)}")
print(f"Attack: {len(attack_indices)}")


# ---------------------------------------------------------
# LEAKAGE-FREE BENIGN SPLIT
# ---------------------------------------------------------
rng = np.random.default_rng(SEED)

shuffled_benign = benign_indices.copy()
rng.shuffle(shuffled_benign)

split = int(
    len(shuffled_benign) * 0.80
)

memory_indices = shuffled_benign[:split]

benign_test_indices = shuffled_benign[split:]

# All attacks are evaluation samples
evaluation_indices = np.concatenate(
    [
        benign_test_indices,
        attack_indices
    ]
)

print("\n" + "=" * 70)
print("LEAKAGE-FREE SPLIT")
print("=" * 70)

print(f"Benign memory bank : {len(memory_indices)}")
print(f"Benign test        : {len(benign_test_indices)}")
print(f"Attack test        : {len(attack_indices)}")
print(f"Total test         : {len(evaluation_indices)}")


# ---------------------------------------------------------
# NORMALIZE EMBEDDINGS
# ---------------------------------------------------------
def l2_normalize(x):

    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True
    )

    return x / np.maximum(
        norms,
        1e-12
    )


memory_embeddings = l2_normalize(
    embeddings[memory_indices]
)

test_embeddings = l2_normalize(
    embeddings[evaluation_indices]
)

test_labels = labels[evaluation_indices]


# ---------------------------------------------------------
# COSINE DISTANCE
# ---------------------------------------------------------
print("\nCalculating cosine kNN scores...")

similarity = (
    test_embeddings
    @ memory_embeddings.T
)

distance = 1.0 - similarity


# ---------------------------------------------------------
# kNN ANOMALY SCORE
# ---------------------------------------------------------
nearest_distances = np.partition(
    distance,
    K - 1,
    axis=1
)[:, :K]

scores = nearest_distances.mean(
    axis=1
)


# ---------------------------------------------------------
# SCORE SUMMARY
# ---------------------------------------------------------
benign_scores = scores[
    test_labels == 0
]

attack_scores = scores[
    test_labels == 1
]

print("\n" + "=" * 70)
print("ANOMALY SCORE SUMMARY")
print("=" * 70)

print("\nBenign:")
print(
    f"Mean   : {benign_scores.mean():.10f}"
)
print(
    f"Median : {np.median(benign_scores):.10f}"
)
print(
    f"95th   : {np.percentile(benign_scores, 95):.10f}"
)
print(
    f"99th   : {np.percentile(benign_scores, 99):.10f}"
)

print("\nAttack:")
print(
    f"Mean   : {attack_scores.mean():.10f}"
)
print(
    f"Median : {np.median(attack_scores):.10f}"
)
print(
    f"95th   : {np.percentile(attack_scores, 95):.10f}"
)
print(
    f"99th   : {np.percentile(attack_scores, 99):.10f}"
)


# ---------------------------------------------------------
# ROC-AUC
# ---------------------------------------------------------
roc_auc = roc_auc_score(
    test_labels,
    scores
)


# ---------------------------------------------------------
# PR-AUC
# ---------------------------------------------------------
pr_auc = average_precision_score(
    test_labels,
    scores
)


print("\n" + "=" * 70)
print("THRESHOLD-FREE PERFORMANCE")
print("=" * 70)

print(
    f"ROC-AUC : {roc_auc:.6f}"
)

print(
    f"PR-AUC  : {pr_auc:.6f}"
)


# ---------------------------------------------------------
# FIND BEST F1 THRESHOLD
# ---------------------------------------------------------
precision, recall, thresholds = precision_recall_curve(
    test_labels,
    scores
)

# precision_recall_curve returns one extra precision/recall
# value compared with thresholds
f1_values = (
    2 * precision[:-1] * recall[:-1]
    /
    np.maximum(
        precision[:-1] + recall[:-1],
        1e-12
    )
)

best_idx = np.argmax(f1_values)

best_threshold = thresholds[best_idx]

best_precision = precision[best_idx]
best_recall = recall[best_idx]
best_f1 = f1_values[best_idx]


# ---------------------------------------------------------
# BEST F1 RESULTS
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)

print(
    f"Threshold : {best_threshold:.10f}"
)

print(
    f"Precision : {best_precision:.6f}"
)

print(
    f"Recall    : {best_recall:.6f}"
)

print(
    f"F1        : {best_f1:.6f}"
)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------
predictions = (
    scores >= best_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    test_labels,
    predictions,
    labels=[0, 1]
).ravel()


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(f"True Negatives  : {tn}")
print(f"False Positives  : {fp}")
print(f"False Negatives  : {fn}")
print(f"True Positives   : {tp}")


# ---------------------------------------------------------
# FINAL METRICS
# ---------------------------------------------------------
final_precision = precision_score(
    test_labels,
    predictions,
    zero_division=0
)

final_recall = recall_score(
    test_labels,
    predictions,
    zero_division=0
)

final_f1 = f1_score(
    test_labels,
    predictions,
    zero_division=0
)

print("\n" + "=" * 70)
print("FINAL DETECTION METRICS")
print("=" * 70)

print(
    f"Precision : {final_precision:.6f}"
)

print(
    f"Recall    : {final_recall:.6f}"
)

print(
    f"F1-score  : {final_f1:.6f}"
)

print(
    f"ROC-AUC   : {roc_auc:.6f}"
)

print(
    f"PR-AUC    : {pr_auc:.6f}"
)


print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "This is an IN-DOMAIN evaluation because "
    "graph_validation.pt is NF-CSE-CIC-IDS2018-v3."
)

print(
    "It is NOT the final cross-dataset zero-day evaluation."
)

print("\nDONE")
print("=" * 70)