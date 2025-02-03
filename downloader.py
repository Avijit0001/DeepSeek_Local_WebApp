from huggingface_hub import snapshot_download
local_model_path = "./deepseek_model"
snapshot_download(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", local_dir=local_model_path)


