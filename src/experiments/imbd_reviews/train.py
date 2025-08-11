import torch
from torch import nn, optim

from .model import IMDbTransformer
from .utils import load_imbd, train_one_epoch, validate
from .early_stopping import EarlyStopping


MAX_LENGTH = 64

D_MODEL = 32
D_FF = 128
H = 4
N_LAYERS = 2

LR = 1e-3
NUM_EPOCHS = 1000
PATIENCE = 5
BATCH_SIZE = 32


def main():
    print("Loading data...")
    train_loader, test_loader, vocab_size = load_imbd(MAX_LENGTH, BATCH_SIZE)
    print("Loading model...")
    model = IMDbTransformer(vocab_size=vocab_size, d_model=D_MODEL, d_ff=D_FF, h=H, n_layers=N_LAYERS, max_seq_len=MAX_LENGTH)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    es = EarlyStopping(patience=PATIENCE, verbose=True, restore_best_weights=True, higher_is_better=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss, train_acc = train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, criterion=criterion, device=device)
        test_loss, test_acc = validate(model=model, dataloader=test_loader, criterion=criterion, device=device)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {test_loss:.4f} | Val Acc: {test_acc:.4f}"
        )

        es(test_loss, model)
        if es.early_stop:
            break


if __name__ == "__main__":
    main()
