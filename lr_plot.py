import matplotlib.pyplot as plt

D_MODEL = 256
WARMUP = 4000

steps = list(range(1, 20001))
lrs = [D_MODEL ** -0.5 * min(s ** -0.5, s * WARMUP ** -1.5) for s in steps]

peak = max(lrs)
print(f"peak lr = {peak:.6f} at step {steps[lrs.index(peak)]}")

plt.figure(figsize=(8, 4.5))
plt.plot(steps, lrs)
plt.axvline(WARMUP, linestyle="--", color="gray", label=f"warmup = {WARMUP} steps")
plt.xlabel("step")
plt.ylabel("learning rate")
plt.title("Learning rate schedule: linear warmup, then step^-0.5 decay")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("results/lr_schedule.png", dpi=150, bbox_inches="tight")
plt.show()