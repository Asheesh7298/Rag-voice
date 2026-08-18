from datasets import load_dataset

print("Checking dataset row counts...")
try:
    ds_hi = load_dataset("ai4bharat/IndicMSMARCO", "hi", split="train")
    print(f"Hindi (hi) total rows: {len(ds_hi)}")
except Exception as e:
    print(f"Hindi error: {e}")

try:
    ds_mr = load_dataset("ai4bharat/IndicMSMARCO", "mr", split="train")
    print(f"Marathi (mr) total rows: {len(ds_mr)}")
except Exception as e:
    print(f"Marathi error: {e}")
