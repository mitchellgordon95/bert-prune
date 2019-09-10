import matplotlib.pyplot as plt
from tables.common import parse_file

FEATURE_COMPARE_TEMPLATE = "{total} / {count} = {cosine}\n"

sparsities = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
differences = []

for sparsity in sparsities:
    cosine = parse_file(f'models/pretrain/gradual_prune_{int(sparsity*100)}/features_compare_0.txt',
                        FEATURE_COMPARE_TEMPLATE)['cosine']
    differences.append(float(cosine))

fig, ax = plt.subplots()
ax.set_title('Average Feature Cosine Sim with Prune 0')
ax.set_xlabel('Prune Percentage')
ax.set_ylabel('Cosine Sim')
ax.set_ylim(0)
ax.plot(sparsities, differences)
fig.savefig('cosine.png')

