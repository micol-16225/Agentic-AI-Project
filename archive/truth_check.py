import pandas as pd
from difflib import get_close_matches

lib = pd.read_csv("master_regulatory_library.csv")
library_sentences = lib['content'].tolist()

# Paste one of your "Hallucinated" quotes here
hallucination = "The use of LOCF for handling missing data is not acceptable"

print(f"Checking: {hallucination}")
matches = get_close_matches(hallucination, library_sentences, n=1, cutoff=0.1)

if matches:
    print(f"Closest Match in Library:\n-> {matches[0]}")
else:
    print("NO MATCH FOUND. The agent invented this entirely.")