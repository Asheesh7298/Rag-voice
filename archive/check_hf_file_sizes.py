from huggingface_hub import HfApi

api = HfApi()
info = api.dataset_info(repo_id="ai4bharat/IndicMSMARCO", files_metadata=True)
print("Files in ai4bharat/IndicMSMARCO:")
for sibling in info.siblings:
    size_mb = round((sibling.size or 0) / (1024*1024), 2)
    print(f"  {sibling.rfilename}: {size_mb} MB")
