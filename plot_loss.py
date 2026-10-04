import json
import matplotlib.pyplot as plt

with open("results/history.json") as f:
    data = json.load(f)

# wrapper kholo: {"params":..., "seconds":..., "history":[...]}
history = data["history"] if isinstance(data, dict) and "history" in data else data

if isinstance(history, list):
    keys = list(history[0].keys())
    print("keys in each epoch record:", keys)
    train_key = next(k for k in keys if "train" in k)
    dev_key = next(k for k in keys if "dev" in k or "val" in k)
    train_loss = [row[train_key] for row in history]
    dev_loss = [row[dev_key] for row in history]
    epochs = [row["epoch"] for row in history] if "epoch" in keys \
        else list(range(1, len(history) + 1))
else:
    print("keys:", list(history.keys()))
    train_key = next(k for k in history if "train" in k)
    dev_key = next(k for k in history if "dev" in k or "val" in k)
    train_loss = history[train_key]
    dev_loss = history[dev_key]
    epochs = list(range(1, len(train_loss) + 1))

best_i = dev_loss.index(min(dev_loss))
best_epoch = epochs[best_i]
print("epochs:", len(epochs), "| best epoch:", best_epoch,
      "| best dev loss:", round(min(dev_loss), 4))

plt.figure(figsize=(8, 4.5))
plt.plot(epochs, train_loss, marker="o", label="train loss")
plt.plot(epochs, dev_loss, marker="o", label="dev loss")
plt.axvline(best_epoch, linestyle="--", color="gray",
            label=f"best epoch ({best_epoch})")
plt.xlabel("epoch")
plt.ylabel("loss (label smoothing 0.1)")
plt.title("Training and dev loss per epoch")
plt.xticks(epochs)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("results/loss_curves.png", dpi=150, bbox_inches="tight")
plt.show()