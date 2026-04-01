import urllib.request
import os

url = "https://github.com/Li-Chongyi/Zero-DCE/raw/master/Zero-DCE_code/snapshots/Epoch99.pth"
save_path = os.path.join(os.path.dirname(__file__), "assets", "Epoch99.pth")

print("Downloading Zero-DCE pre-trained weights to yield empirical results...")
urllib.request.urlretrieve(url, save_path)
print(f"Downloaded successfully to {save_path}")
