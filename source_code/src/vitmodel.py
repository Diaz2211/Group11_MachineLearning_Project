import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 1. Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang chạy trên thiết bị: {device}")

dataset_path = "source_code/Data"

# 2. Khởi tạo Feature Extractor của ViT
model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTImageProcessor.from_pretrained(model_name)

# Tiền xử lý ảnh theo chuẩn mã nguồn của ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# 3. Đọc và chia Dataset (80% Train, 20% Val)
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

train_idx, val_idx = train_test_split(
    list(range(len(full_dataset))), test_size=0.2, random_state=42, stratify=full_dataset.targets
)

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 4. Tải mô hình ViT pre-trained và cấu hình phân loại 38 lớp
model = ViTForImageClassification.from_pretrained(
    model_name, 
    num_labels=38,
    id2label={str(i): str(i) for i in range(38)},
    label2id={str(i): i for i in range(38)}
)
model.to(device)

# 5. Cài đặt Loss và Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 6. Vòng lặp Huấn luyện (Train Loop)
epochs = 5
best_acc = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    # Đánh giá trên tập Validation sau mỗi Epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
    val_acc = correct / total
    print(f"Loss: {running_loss/len(train_loader):.4f} | Val Accuracy: {val_acc*100:.2f}%")
    
    # Lưu mô hình tốt nhất
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_vit_model.pt')
        print("=> Đã lưu mô hình ViT tốt nhất!")
