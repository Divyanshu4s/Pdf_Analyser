from huggingface_hub import snapshot_download

# Which model to grab
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Where to place the files
snapshot_download(
    repo_id=MODEL_ID,
    local_dir="model",                 # download directly into ./model
    local_dir_use_symlinks=False,      # make real copies instead of symlinks
    allow_patterns=[                   # grab only the files you need
        "pytorch_model.bin",           # ← weights (~120 MB)
        "*.json",                      # configs + tokenizer files
        "*.txt"                        # vocab.txt if present
    ]
)

print("✅  Model downloaded to ./model")
