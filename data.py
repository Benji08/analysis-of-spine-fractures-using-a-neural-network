import os
import re
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt

# Parameters
FOLDER_PATH = "C:/Users/barba/OneDrive/Pulpit/data_set"
pattern = re.compile(r'^(T|L)(\d{2})-((A\d{1})|__)-((B\d{1}|C|__){1,3})\.nrrd$')

vertebra_counts = Counter()
fracture_counts = Counter()

all_vertebrae = [f"L{str(i).zfill(2)}" for i in reversed(range(1, 6))] + [f"T{str(i).zfill(2)}" for i in reversed(range(1, 13))]
all_fractures = ['A0', 'A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'C']

# Counting
for subdir, dirs, files in os.walk(FOLDER_PATH):
    for file in files:
        if file.endswith(".nrrd"):
            match = pattern.match(file)
            if match:
                segment = match.group(1)
                number = match.group(2)
                a_type = match.group(3)
                bc_type = match.group(5)

                vertebra = f"{segment}{number}"
                vertebra_counts[vertebra] += 1

                if a_type == "__" and bc_type == "__":
                    fracture_counts["ZDROWY"] += 1
                else:
                    if a_type != "__":
                        fracture_counts[a_type] += 1
                    if bc_type != "__" and bc_type != "C1":
                        fracture_counts[bc_type] += 1

for v in all_vertebrae:
    vertebra_counts.setdefault(v, 0)
for f in all_fractures:
    fracture_counts.setdefault(f, 0)

# Results
healthy_count = fracture_counts.get("ZDROWY", 0)
print(f"\n🟩 Liczba zdrowych kręgów: {healthy_count}\n")

vertebra_data = OrderedDict((v, vertebra_counts[v]) for v in all_vertebrae)
fracture_data = OrderedDict((f, fracture_counts[f]) for f in all_fractures)

print("🦴 Liczba przykładów dla każdego kręgu:")
for v, count in vertebra_data.items():
    print(f"  {v}: {count}")

print("\n💥 Liczba przykładów dla każdego typu złamania:")
for f, count in fracture_data.items():
    print(f"  {f}: {count}")

# Plot 1 – number of types of vertebrae
plt.figure(figsize=(12, 5))
plt.bar(vertebra_data.keys(), vertebra_data.values(), color='skyblue')
plt.title('Liczba przykładów dla każdego kręgu (od dołu kręgosłupa)')
plt.xlabel('Krąg')
plt.ylabel('Liczba przykładów')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Plot 2 – number of types of vertebral fractures
plt.figure(figsize=(12, 5))
plt.bar(fracture_data.keys(), fracture_data.values(), color='salmon')
plt.title('Liczba przykładów dla każdego rodzaju złamania')
plt.xlabel('Rodzaj złamania')
plt.ylabel('Liczba przykładów')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
