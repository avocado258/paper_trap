import pandas as pd

real = pd.read_csv('../../data/real_papers.csv')
fake = pd.read_csv('../../data/fake_papers.csv')

# Keep only text and label
real = real[['text', 'label']]
fake = fake[['text', 'label']]

# Combine and shuffle
dataset = pd.concat([real, fake], ignore_index=True)
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
dataset.to_csv('../../data/final_dataset.csv', index=False)

print(f"Final dataset: {len(dataset)} papers")
print(dataset['label'].value_counts())