import matplotlib.pyplot as plt

# Input data + 2 termination bits
data = [1, 0, 1, 1, 0, 0]

# Convolutional encoder (2,1,2)
m1 = m2 = 0
encoded = []

for b in data:
    c1 = b ^ m1 ^ m2
    c2 = b ^ m2

    encoded += [c1, c2]

    m2, m1 = m1, b

print("Input   :", data)
print("Encoded :", encoded)

# -------- Graph --------
plt.figure(figsize=(9, 5))

plt.subplot(2, 1, 1)
plt.step(range(len(data)), data, where="post")
plt.title("Input Binary Data")
plt.ylabel("Bit")
plt.grid()

plt.subplot(2, 1, 2)
plt.step(range(len(encoded)), encoded, where="post")
plt.title("Convolutionally Encoded Data")
plt.xlabel("Encoded Bit Number")
plt.ylabel("Bit")
plt.grid()

plt.tight_layout()
plt.show()