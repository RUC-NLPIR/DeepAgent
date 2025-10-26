import os
import json

src_dir = "/mnt/ali-sh-1/usr/tusen/xiaoxi/DeepAgent/data/API-Bank/lv1-lv2-samples/level-1-given-desc"
dst_dir = "/mnt/ali-sh-1/usr/tusen/xiaoxi/DeepAgent/data/API-Bank/lv1-lv2-samples/level-1-given-desc-e2e"

os.makedirs(dst_dir, exist_ok=True)

src_files = [fname for fname in os.listdir(src_dir) if fname.endswith(".jsonl")]
total_files = len(src_files)
saved_files = 0

for fname in src_files:
    src_path = os.path.join(src_dir, fname)
    with open(src_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    roles = []
    for line in lines:
        try:
            obj = json.loads(line)
            roles.append(obj.get("role", ""))
        except Exception:
            roles.append("")
    # Find the last "User" (case-insensitive, but likely "User")
    last_user_idx = -1
    for idx, role in enumerate(roles):
        if role.lower() == "user":
            last_user_idx = idx
    # Check if any "API" appears before or at last_user_idx
    has_api_before_last_user = False
    for idx, role in enumerate(roles[:last_user_idx+1]):
        if role == "API":
            has_api_before_last_user = True
            break
    if not has_api_before_last_user:
        # Save to dst_dir
        dst_path = os.path.join(dst_dir, fname)
        with open(dst_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        saved_files += 1

print(f"源文件夹有 {total_files} 个文件，保存的有 {saved_files} 个文件")
