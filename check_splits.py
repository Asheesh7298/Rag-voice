from datasets import get_dataset_config_names, get_dataset_split_names

configs = get_dataset_config_names("ai4bharat/IndicMSMARCO")
print("Available configs in IndicMSMARCO:", configs)

for c in ["hi", "mr"]:
    splits = get_dataset_split_names("ai4bharat/IndicMSMARCO", c)
    print(f"Splits for {c}:", splits)
