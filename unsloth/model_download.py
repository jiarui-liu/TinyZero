# Load model directly
import sys

from huggingface_hub import snapshot_download

# Set your model repository name (find it on Hugging Face)
model_repo = "unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF"  # Change this to your model
subfolder = "DeepSeek-R1-Distill-Llama-70B-Q8_0"
local_dir = sys.argv[1]  # Change to your desired local path

# Download the entire model repository
# snapshot_download(repo_id=model_repo, local_dir=local_dir)
snapshot_download(
    repo_id=model_repo,
    local_dir=local_dir,
    allow_patterns=[f"{subfolder}/*"],  # Only download files inside this subfolder
)

