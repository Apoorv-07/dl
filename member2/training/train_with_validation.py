import sys
import os
import json

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


# ============================================================
# IMPORTS
# ============================================================

import torch
from torch.optim import Adam

from models.gatv2_encoder import GATv2Encoder
from models.temporal_transformer import TemporalTransformer
from models.projection_head import ProjectionHead

from training.augmentations import create_two_views
from losses.nt_xent import NTXentLoss
from models.global_pooling import global_mean_pooling


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32

# Maximum training duration.
# Early stopping may terminate training before this.
MAX_EPOCHS = 100

# Early stopping configuration.
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 1e-4

LEARNING_RATE = 1e-4

DROPOUT_PROBABILITY = 0.5
FEATURE_MASK_PROBABILITY = 0.15

NT_XENT_TEMPERATURE = 0.1

# None = use complete Dataset 1.
MAX_SEQUENCES = None

TRAIN_SPLIT = 0.90

SEED = 42


# ============================================================
# CHECKPOINT DIRECTORIES
# ============================================================

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints_validation"
)

HISTORY_PATH = os.path.join(
    CHECKPOINT_DIR,
    "training_history.json"
)

BEST_CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pt"
)

# ============================================================
# RESUME CONFIGURATION
# ============================================================

# Set to False to run a clean training pass with fixed loss & augmentations
RESUME_TRAINING = False

RESUME_CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "epoch_2.pt"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# ENCODE ONE SEQUENCE
# ============================================================

def encode_sequence(
    sequence,
    gat,
    transformer,
    projection
):

    snapshot_embeddings = []

    for graph in sequence:

        x = graph.x.to(DEVICE) if graph.x is not None else None

        edge_index = graph.edge_index.to(DEVICE) if graph.edge_index is not None else None

        edge_attr = graph.edge_attr.to(DEVICE) if graph.edge_attr is not None else None

        # ----------------------------------------------------
        # Clamp continuous features to [-5.0, 5.0]
        # ----------------------------------------------------

        if x is not None:
            x = torch.clamp(x, min=-5.0, max=5.0)

        if edge_attr is not None:
            edge_attr = torch.clamp(edge_attr, min=-5.0, max=5.0)

        # ----------------------------------------------------
        # GATv2
        # ----------------------------------------------------

        node_embeddings = gat(
            x,
            edge_index,
            edge_attr
        )

        # ----------------------------------------------------
        # Global Mean Pooling
        # ----------------------------------------------------

        graph_embedding = global_mean_pooling(
            node_embeddings
        )

        snapshot_embeddings.append(
            graph_embedding
        )

    # --------------------------------------------------------
    # [5, 128]
    # --------------------------------------------------------

    temporal_input = torch.cat(
        snapshot_embeddings,
        dim=0
    )

    # --------------------------------------------------------
    # [1, 5, 128]
    # --------------------------------------------------------

    temporal_input = temporal_input.unsqueeze(0)

    # --------------------------------------------------------
    # Temporal Transformer
    # --------------------------------------------------------

    h = transformer(
        temporal_input
    )

    # --------------------------------------------------------
    # Projection Head
    # --------------------------------------------------------

    z = projection(h)

    return z


# ============================================================
# RUN ONE TRAINING EPOCH
# ============================================================

def train_one_epoch(
    train_sequences,
    gat,
    transformer,
    projection,
    criterion,
    optimizer
):

    gat.train()
    transformer.train()
    projection.train()

    epoch_loss = 0.0

    permutation = torch.randperm(
        len(train_sequences)
    ).tolist()

    shuffled_sequences = [
        train_sequences[i]
        for i in permutation
    ]

    num_batches = (
        len(shuffled_sequences)
        // BATCH_SIZE
    )

    for batch_idx in range(num_batches):

        batch_sequences = shuffled_sequences[
            batch_idx * BATCH_SIZE:
            (batch_idx + 1) * BATCH_SIZE
        ]

        z1_list = []
        z2_list = []

        for item in batch_sequences:

            snapshots = item["snapshots"]

            # ------------------------------------------------
            # Two stochastic augmented views
            # ------------------------------------------------

            view1, view2 = create_two_views(
                snapshots,
                dropout_probability=DROPOUT_PROBABILITY,
                feature_mask_probability=FEATURE_MASK_PROBABILITY
            )

            # ------------------------------------------------
            # Encode view 1
            # ------------------------------------------------

            z1 = encode_sequence(
                view1,
                gat,
                transformer,
                projection
            )

            # ------------------------------------------------
            # Encode view 2
            # ------------------------------------------------

            z2 = encode_sequence(
                view2,
                gat,
                transformer,
                projection
            )

            z1_list.append(z1)

            z2_list.append(z2)

        # ----------------------------------------------------
        # [BATCH_SIZE, 64]
        # ----------------------------------------------------

        z1_batch = torch.cat(
            z1_list,
            dim=0
        )

        z2_batch = torch.cat(
            z2_list,
            dim=0
        )

        # ----------------------------------------------------
        # NT-Xent
        # ----------------------------------------------------

        loss = criterion(
            z1_batch,
            z2_batch
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        print(
            f"Batch {batch_idx + 1}/{num_batches} "
            f"| Loss: {loss.item():.6f}"
        )

    average_loss = (
        epoch_loss / num_batches
    )

    return average_loss


# ============================================================
# VALIDATION
# ============================================================

def validate(
    val_sequences,
    gat,
    transformer,
    projection,
    criterion
):

    gat.eval()
    transformer.eval()
    projection.eval()

    validation_loss = 0.0

    num_batches = (
        len(val_sequences)
        // BATCH_SIZE
    )

    with torch.no_grad():

        for batch_idx in range(num_batches):

            batch_sequences = val_sequences[
                batch_idx * BATCH_SIZE:
                (batch_idx + 1) * BATCH_SIZE
            ]

            z1_list = []
            z2_list = []

            for item in batch_sequences:

                snapshots = item["snapshots"]

                # ------------------------------------------------
                # Validation augmented views
                # ------------------------------------------------

                view1, view2 = create_two_views(
                    snapshots,
                    dropout_probability=DROPOUT_PROBABILITY,
                    feature_mask_probability=FEATURE_MASK_PROBABILITY
                )

                # ------------------------------------------------
                # Encode both views
                # ------------------------------------------------

                z1 = encode_sequence(
                    view1,
                    gat,
                    transformer,
                    projection
                )

                z2 = encode_sequence(
                    view2,
                    gat,
                    transformer,
                    projection
                )

                z1_list.append(z1)

                z2_list.append(z2)

            z1_batch = torch.cat(
                z1_list,
                dim=0
            )

            z2_batch = torch.cat(
                z2_list,
                dim=0
            )

            loss = criterion(
                z1_batch,
                z2_batch
            )

            validation_loss += loss.item()

    average_validation_loss = (
        validation_loss / num_batches
    )

    return average_validation_loss


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    epoch,
    gat,
    transformer,
    projection,
    optimizer,
    train_loss,
    validation_loss,
    best_validation_loss,
    best_epoch,
    patience_counter
):

    torch.save(
        {
            "epoch": epoch,

            "gat_state_dict":
                gat.state_dict(),

            "transformer_state_dict":
                transformer.state_dict(),

            "projection_state_dict":
                projection.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_loss":
                train_loss,

            "validation_loss":
                validation_loss,

            "best_validation_loss":
                best_validation_loss,

            "best_epoch":
                best_epoch,

            "patience_counter":
                patience_counter,

            "max_epochs":
                MAX_EPOCHS,

            "early_stopping_patience":
                EARLY_STOPPING_PATIENCE,

            "min_delta":
                MIN_DELTA,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "nt_xent_temperature":
                NT_XENT_TEMPERATURE,

            "augmentation_dropout_probability":
                DROPOUT_PROBABILITY,

            "augmentation_feature_mask_probability":
                FEATURE_MASK_PROBABILITY,

            "train_split":
                TRAIN_SPLIT,

            "seed":
                SEED,
        },
        path
    )


# ============================================================
# MAIN
# ============================================================

print("======================================")
print("MEMBER 2 SELF-SUPERVISED TRAINING")
print("WITH EPOCH-WISE VALIDATION")
print("EARLY STOPPING ENABLED")
print("======================================")


# ============================================================
# TRAINING CONFIGURATION DISPLAY
# ============================================================

print()
print("Device:", DEVICE)

if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()
print("Maximum epochs:", MAX_EPOCHS)

print(
    "Early stopping patience:",
    EARLY_STOPPING_PATIENCE
)

print(
    "Minimum validation improvement:",
    MIN_DELTA
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "NT-Xent temperature:",
    NT_XENT_TEMPERATURE
)

print(
    "Augmentation dropout:",
    DROPOUT_PROBABILITY
)

print(
    "Augmentation feature mask:",
    FEATURE_MASK_PROBABILITY
)

print(
    "Resume training:",
    RESUME_TRAINING
)

if RESUME_TRAINING:
    print(
        "Resume checkpoint:",
        RESUME_CHECKPOINT_PATH
    )


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("Loading graph.pt...")

data = torch.load(
    os.path.join(
        PROJECT_ROOT,
        "graph.pt"
    ),
    weights_only=False
)

sequences = data["sequences"]

print(
    "Total sequences:",
    len(sequences)
)


# ============================================================
# DATASET SELECTION
# ============================================================

if MAX_SEQUENCES is None:

    selected_indices = list(
        range(len(sequences))
    )

else:

    generator = torch.Generator().manual_seed(
        SEED
    )

    indices = torch.randperm(
        len(sequences),
        generator=generator
    ).tolist()

    selected_indices = indices[
        :MAX_SEQUENCES
    ]


selected_sequences = [
    sequences[i]
    for i in selected_indices
]


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

generator = torch.Generator().manual_seed(
    SEED
)

permutation = torch.randperm(
    len(selected_sequences),
    generator=generator
).tolist()

split_index = int(
    TRAIN_SPLIT *
    len(selected_sequences)
)

train_indices = permutation[
    :split_index
]

val_indices = permutation[
    split_index:
]

train_sequences = [
    selected_sequences[i]
    for i in train_indices
]

val_sequences = [
    selected_sequences[i]
    for i in val_indices
]


print()
print("======================================")
print("DATASET SPLIT")
print("======================================")

print(
    "Training sequences:",
    len(train_sequences)
)

print(
    "Validation sequences:",
    len(val_sequences)
)


# ============================================================
# CREATE MODELS
# ============================================================

print()
print("Creating models...")

gat = GATv2Encoder(
    in_channels=53,
    hidden_channels=128,
    heads=4,
    dropout=0.2
).to(DEVICE)


transformer = TemporalTransformer(
    input_dim=128,
    num_heads=4,
    num_layers=2,
    dropout=0.1
).to(DEVICE)


projection = ProjectionHead(
    input_dim=128,
    hidden_dim=128,
    output_dim=64
).to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = NTXentLoss(
    temperature=NT_XENT_TEMPERATURE
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = Adam(
    list(gat.parameters())
    + list(transformer.parameters())
    + list(projection.parameters()),
    lr=LEARNING_RATE
)


# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# TRAINING HISTORY / RESUME STATE
# ============================================================

history = []

best_validation_loss = float("inf")

best_epoch = None

patience_counter = 0

start_epoch = 1


if RESUME_TRAINING:

    print()
    print("======================================")
    print("RESUMING TRAINING")
    print("======================================")

    print()
    print(
        "Resume checkpoint:",
        RESUME_CHECKPOINT_PATH
    )

    if not os.path.exists(
        RESUME_CHECKPOINT_PATH
    ):
        raise FileNotFoundError(
            "Resume checkpoint not found: "
            + RESUME_CHECKPOINT_PATH
        )

    resume_checkpoint = torch.load(
        RESUME_CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    required_keys = [
        "epoch",
        "gat_state_dict",
        "transformer_state_dict",
        "projection_state_dict",
        "optimizer_state_dict",
        "train_loss",
        "validation_loss",
        "best_validation_loss",
        "best_epoch",
        "patience_counter"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in resume_checkpoint
    ]

    if missing_keys:
        raise RuntimeError(
            "Resume checkpoint is missing required keys: "
            + str(missing_keys)
        )

    completed_epoch = int(
        resume_checkpoint["epoch"]
    )

    if completed_epoch >= MAX_EPOCHS:

        raise RuntimeError(
            "Resume checkpoint is already at or beyond "
            "MAX_EPOCHS. Nothing to resume."
        )

    print()
    print(
        "Completed epoch:",
        completed_epoch
    )

    print(
        "Checkpoint training loss:",
        resume_checkpoint["train_loss"]
    )

    print(
        "Checkpoint validation loss:",
        resume_checkpoint["validation_loss"]
    )

    print(
        "Checkpoint best validation loss:",
        resume_checkpoint["best_validation_loss"]
    )

    print(
        "Checkpoint best epoch:",
        resume_checkpoint["best_epoch"]
    )

    print(
        "Checkpoint patience:",
        resume_checkpoint["patience_counter"]
    )

    # --------------------------------------------------------
    # Restore model weights
    # --------------------------------------------------------

    gat.load_state_dict(
        resume_checkpoint["gat_state_dict"]
    )

    transformer.load_state_dict(
        resume_checkpoint["transformer_state_dict"]
    )

    projection.load_state_dict(
        resume_checkpoint["projection_state_dict"]
    )

    # --------------------------------------------------------
    # Restore optimizer state
    # --------------------------------------------------------

    optimizer.load_state_dict(
        resume_checkpoint["optimizer_state_dict"]
    )

    # --------------------------------------------------------
    # Restore validation / early-stopping state
    # --------------------------------------------------------

    best_validation_loss = float(
        resume_checkpoint["best_validation_loss"]
    )

    best_epoch = int(
        resume_checkpoint["best_epoch"]
    )

    patience_counter = int(
        resume_checkpoint["patience_counter"]
    )

    # --------------------------------------------------------
    # Restore previous history
    # --------------------------------------------------------

    if os.path.exists(HISTORY_PATH):

        with open(
            HISTORY_PATH,
            "r"
        ) as f:

            history_data = json.load(f)

        previous_history = history_data.get(
            "epochs",
            []
        )

        # Keep only completed epochs through the resume point.
        history = [
            entry
            for entry in previous_history
            if int(entry["epoch"]) <= completed_epoch
        ]

        if len(history) != completed_epoch:

            raise RuntimeError(
                "training_history.json does not contain a "
                "complete history through resume epoch "
                f"{completed_epoch}."
            )

        history_best_epoch = history_data.get(
            "best_epoch"
        )

        history_best_validation_loss = history_data.get(
            "best_validation_loss"
        )

        if history_best_epoch != best_epoch:
            raise RuntimeError(
                "Checkpoint/history best_epoch mismatch: "
                f"checkpoint={best_epoch}, "
                f"history={history_best_epoch}"
            )

        if abs(
            float(history_best_validation_loss)
            - best_validation_loss
        ) > 1e-12:

            raise RuntimeError(
                "Checkpoint/history best_validation_loss mismatch: "
                f"checkpoint={best_validation_loss}, "
                f"history={history_best_validation_loss}"
            )

    else:

        raise FileNotFoundError(
            "Cannot resume safely because training history is missing: "
            + HISTORY_PATH
        )

    start_epoch = completed_epoch + 1

    print()
    print(
        "History entries restored:",
        len(history)
    )

    print(
        "Starting from epoch:",
        start_epoch
    )

    print(
        "Will train through epoch:",
        MAX_EPOCHS
    )

else:

    print()
    print("Starting a new training run.")

    start_epoch = 1


# ============================================================
# TRAINING
# ============================================================

print()
print("======================================")
print("STARTING TRAINING")
print("======================================")

for epoch in range(start_epoch - 1, MAX_EPOCHS):

    current_epoch = epoch + 1

    print()
    print("======================================")
    print(
        f"EPOCH {current_epoch}/{MAX_EPOCHS}"
    )
    print("======================================")


    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("TRAINING")
    print()

    train_loss = train_one_epoch(
        train_sequences,
        gat,
        transformer,
        projection,
        criterion,
        optimizer
    )

    print()
    print("--------------------------------------")

    print(
        f"Average training loss: "
        f"{train_loss:.6f}"
    )

    print("--------------------------------------")


    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("VALIDATION")
    print()

    validation_loss = validate(
        val_sequences,
        gat,
        transformer,
        projection,
        criterion
    )

    print()
    print("--------------------------------------")

    print(
        f"Validation loss: "
        f"{validation_loss:.6f}"
    )

    print("--------------------------------------")


    # ========================================================
    # DETERMINE VALIDATION IMPROVEMENT
    # ========================================================

    improvement = (
        best_validation_loss -
        validation_loss
    )

    is_best = (
        validation_loss <
        best_validation_loss - MIN_DELTA
    )


    # ========================================================
    # UPDATE BEST MODEL / PATIENCE
    # ========================================================

    if is_best:

        best_validation_loss = (
            validation_loss
        )

        best_epoch = current_epoch

        patience_counter = 0

        print()
        print("NEW BEST MODEL")

        print(
            "Best validation loss:",
            f"{best_validation_loss:.6f}"
        )

        print(
            "Improvement:",
            f"{improvement:.6f}"
        )

        # ----------------------------------------------------
        # Save dedicated BEST checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            BEST_CHECKPOINT_PATH,
            current_epoch,
            gat,
            transformer,
            projection,
            optimizer,
            train_loss,
            validation_loss,
            best_validation_loss,
            best_epoch,
            patience_counter
        )

        print()
        print(
            "Best checkpoint saved:",
            BEST_CHECKPOINT_PATH
        )

    else:

        patience_counter += 1

        print()
        print(
            "No meaningful validation improvement."
        )

        print(
            "Patience:",
            f"{patience_counter}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )


    # ========================================================
    # HISTORY
    # ========================================================

    history_entry = {

        "epoch":
            current_epoch,

        "train_loss":
            train_loss,

        "validation_loss":
            validation_loss,

        "best_validation_loss":
            best_validation_loss,

        "best_epoch":
            best_epoch,

        "patience_counter":
            patience_counter
    }

    history.append(
        history_entry
    )


    # ========================================================
    # SAVE EPOCH CHECKPOINT
    # ========================================================

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"epoch_{current_epoch}.pt"
    )

    save_checkpoint(
        checkpoint_path,
        current_epoch,
        gat,
        transformer,
        projection,
        optimizer,
        train_loss,
        validation_loss,
        best_validation_loss,
        best_epoch,
        patience_counter
    )

    print()
    print(
        "Epoch checkpoint saved:",
        checkpoint_path
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    with open(
        HISTORY_PATH,
        "w"
    ) as f:

        json.dump(
            {
                "configuration": {

                    "max_epochs":
                        MAX_EPOCHS,

                    "early_stopping_patience":
                        EARLY_STOPPING_PATIENCE,

                    "min_delta":
                        MIN_DELTA,

                    "batch_size":
                        BATCH_SIZE,

                    "learning_rate":
                        LEARNING_RATE,

                    "nt_xent_temperature":
                        NT_XENT_TEMPERATURE,

                    "augmentation_dropout_probability":
                        DROPOUT_PROBABILITY,

                    "augmentation_feature_mask_probability":
                        FEATURE_MASK_PROBABILITY,

                    "train_split":
                        TRAIN_SPLIT,

                    "seed":
                        SEED
                },

                "epochs":
                    history,

                "best_epoch":
                    best_epoch,

                "best_validation_loss":
                    best_validation_loss
            },
            f,
            indent=4
        )

    print()
    print(
        "History saved:",
        HISTORY_PATH
    )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if patience_counter >= EARLY_STOPPING_PATIENCE:

        print()
        print("======================================")
        print("EARLY STOPPING TRIGGERED")
        print("======================================")

        print()
        print(
            "No meaningful validation improvement "
            f"for {EARLY_STOPPING_PATIENCE} consecutive epochs."
        )

        print(
            "Best epoch:",
            best_epoch
        )

        print(
            "Best validation loss:",
            f"{best_validation_loss:.6f}"
        )

        break


# ============================================================
# TRAINING COMPLETE
# ============================================================

print()
print("======================================")
print("TRAINING COMPLETE")
print("======================================")

print()

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation loss:",
    f"{best_validation_loss:.6f}"
)

print()

print(
    "Best checkpoint:",
    BEST_CHECKPOINT_PATH
)

print(
    "Training history:",
    HISTORY_PATH
)


# ============================================================
# FINAL LOSS HISTORY
# ============================================================

print()
print("======================================")
print("FINAL LOSS HISTORY")
print("======================================")

for entry in history:

    print(
        f"Epoch {entry['epoch']} | "
        f"Train: {entry['train_loss']:.6f} | "
        f"Validation: {entry['validation_loss']:.6f} | "
        f"Best: {entry['best_validation_loss']:.6f} | "
        f"Patience: {entry['patience_counter']}"
    )


print()
print("======================================")
print("VALIDATION TRAINING COMPLETE")
print("======================================")