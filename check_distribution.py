import pandas as pd

# Check current splits
train = pd.read_csv('processed/train.csv')
val = pd.read_csv('processed/validation.csv')
test = pd.read_csv('processed/test.csv')

print('=== Current Data Distribution ===')
print(f'Train: {len(train)} samples')
print(f'  Safe: {(train["label"] == 0).sum()}')
print(f'  Bullying: {(train["label"] == 1).sum()}')
print()
print(f'Val: {len(val)} samples')
print(f'  Safe: {(val["label"] == 0).sum()}')
print(f'  Bullying: {(val["label"] == 1).sum()}')
print()
print(f'Test: {len(test)} samples')
print(f'  Safe: {(test["label"] == 0).sum()}')
print(f'  Bullying: {(test["label"] == 1).sum()}')
