import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer('all-MiniLM-L6-v2')
stat_df = pd.read_csv("statutory_truth_with_ids.csv")
intel_df = pd.read_csv("optimizer_intelligence_with_ids.csv")

print("📦 Baking embeddings inside Docker...")
torch.save(model.encode(intel_df[intel_df['type'] == 'Academic_Rigor']['content'].fillna("").tolist(), convert_to_tensor=True), "acad_embeddings.pt")
torch.save(model.encode(stat_df[stat_df['type'] == 'Precedent']['content'].fillna("").tolist(), convert_to_tensor=True), "letter_embeddings.pt")
torch.save(model.encode(stat_df[stat_df['type'] == 'Statutory']['content'].fillna("").tolist(), convert_to_tensor=True), "stat_law_embeddings.pt")
print("✅ Done.")