import matplotlib.pyplot as plt
from embeddings import PositionalEncoding

layer = PositionalEncoding(256)
matrix = layer.pe[0, :100, :].numpy()    

# 3) Heat-map banao
plt.figure(figsize=(10, 5))
plt.imshow(matrix, aspect="auto", cmap="viridis")
plt.colorbar(label="PE value")
plt.xlabel("Embedding dimension (0-255)")
plt.ylabel("Position (0-99)")
plt.title("Sinusoidal positional encoding")
plt.savefig("../results/pe_heatmap.png", dpi=150)
plt.show()