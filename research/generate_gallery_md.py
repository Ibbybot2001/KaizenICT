
import os
import glob

base_dir = r"C:\Users\CEO\.gemini\antigravity\brain\26dbde5c-a7fe-4c77-bbbc-7c4ff10a53fd"
gallery_path = os.path.join(base_dir, "chart_gallery.md")

def get_images(pattern):
    files = glob.glob(os.path.join(base_dir, pattern))
    # Sort by filename
    files.sort(key=lambda x: os.path.basename(x))
    return files

wins = get_images("win_ex_*.png")
losses = get_images("loss_ex_*.png")
rejs = get_images("rej_ex_*.png")

def make_carousel(images, title):
    md = []
    for img in images:
        # Use simple filenames if in same dir, or absolute paths
        # Artifacts require absolute paths in this environment usually, 
        # but let's stick to the protocol: ![Caption](/abs/path)
        path = img.replace("\\", "/") # Ensure forward slashes
        name = os.path.basename(img)
        md.append(f"![{name}](/{path})")
        md.append("<!-- slide -->")
    
    # Remove last slide separator
    if md and md[-1] == "<!-- slide -->":
        md.pop()
        
    return "\n".join(md)

content = []
content.append("# Chart Gallery\n")

content.append("## 🚫 20 Rejected Trades (Randomized)")
if rejs:
    content.append("````carousel")
    content.append(make_carousel(rejs, "Rejected"))
    content.append("````")
else:
    content.append("_No images found_")

content.append("\n## ❌ 20 Losses (Randomized)")
if losses:
    content.append("````carousel")
    content.append(make_carousel(losses, "Loss"))
    content.append("````")
else:
    content.append("_No images found_")

content.append("\n## ✅ 20 Wins (Randomized)")
if wins:
    content.append("````carousel")
    content.append(make_carousel(wins, "Win"))
    content.append("````")
else:
    content.append("_No images found_")

with open(gallery_path, "w", encoding="utf-8") as f:
    f.write("\n".join(content))

print(f"Generated {gallery_path} with {len(wins)} wins, {len(losses)} losses, {len(rejs)} rejections.")
