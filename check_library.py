import pandas as pd

# Load the library
file_path = 'master_regulatory_library.csv'

try:
    df = pd.read_csv(file_path)
    # Clean column names in case of trailing spaces
    df.columns = [c.strip() for c in df.columns]

    print("--- 📚 SOURCE OF TRUTH SUMMARY ---")
    print(f"Total Rows: {len(df)}")
    print(df['type'].value_counts())
    print("-" * 34)

    # Search for ICH E3 and ICH E9 specifically
    targets = ['ICH E3', 'ICH E9']
    
    for target in targets:
        print(f"\n🔍 EXACT QUOTES FOR: {target}")
        # Searching in the citation/source column
        matches = df[df.apply(lambda row: target in str(row.values), axis=1)]
        
        if matches.empty:
            print(f"   ❌ No exact matches found for {target}.")
        else:
            for i, row in matches.iterrows():
                # Including Statutory Type as per Jan 2 Directive
                print(f"[{row['type'].upper()}] | {row['content'][:150]}...")
                print(f"   Full Citation: {row.get('citation_source', 'N/A')}\n")

except FileNotFoundError:
    print(f"❌ Error: {file_path} not found in this directory.")