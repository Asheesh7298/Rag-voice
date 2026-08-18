from huggingface_hub import HfApi, list_repo_files

api = HfApi()
print("Listing files in ai4bharat/IndicMSMARCO...")
try:
    files = list_repo_files(repo_id="ai4bharat/IndicMSMARCO", repo_type="dataset")
    print(f"Total files in repo: {len(files)}")
    for f in files[:30]:
        print("  ", f)
except Exception as e:
    print("Error:", e)
