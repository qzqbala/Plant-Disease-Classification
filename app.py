import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Classification")
st.write("Upload a plant leaf image, and the model will predict its disease class.")

# CustomCNN с BatchNorm
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.5)

        with torch.no_grad():
            dummy = torch.zeros(1,3,224,224)
            x = self.pool(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            flatten_dim = x.numel()

        self.fc1 = nn.Linear(flatten_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x))); x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Classes (топ-8)

classes = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___healthy",
    "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
    "Tomato_Bacterial_spot", "Tomato_Late_blight", "Tomato__Tomato_YellowLeaf__Curl_Virus"
]
num_classes = len(classes)

# Трансформация для инференса
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# Загрузка модели
@st.cache_resource
def load_model():
    model = CustomCNN(num_classes).to(DEVICE)
    state_dict = torch.load("best_model_filtered.pth", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()


# Upload & inference
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg","jpeg","png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        input_tensor = val_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)

        # Top-1
        confidence, predicted_class = torch.max(probs, 1)
        pred_label = classes[predicted_class.item()]
        confidence_score = confidence.item()*100

        # Top-3
        top_probs, top_idx = torch.topk(probs, 3)
        top_classes = [classes[i] for i in top_idx[0]]
        top_conf = top_probs[0].cpu().numpy()*100

    st.subheader(f"Predicted: **{pred_label}**")
    st.progress(confidence.item())
    st.caption(f"Confidence: {confidence_score:.2f}%")

    st.write("Top-3 predictions:")
    for i in range(3):
        st.write(f"{top_classes[i]}: {top_conf[i]:.2f}%")
