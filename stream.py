import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import os
from collections import defaultdict
from io import BytesIO
import cv2
import tempfile

# -----------------------
# Device
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Vocabulary class
# -----------------------
class Vocabulary:
    def __init__(self, word2idx=None, idx2word=None):
        self.word2idx = word2idx
        self.idx2word = idx2word

# -----------------------
# Encoder
# -----------------------
class EncoderCNN(torch.nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = torch.nn.Linear(self.cnn.fc.in_features, embed_size)
        self.bn = torch.nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.cnn(images)
        features = self.bn(features)
        return features.unsqueeze(1)

# -----------------------
# Decoder
# -----------------------
class TransformerDecoder(torch.nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads=8, num_layers=3, forward_expansion=2048, dropout=0.1, max_len=50):
        super().__init__()
        self.embed_size = embed_size
        self.word_embedding = torch.nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = torch.nn.Embedding(max_len, embed_size)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads,
                                                        dim_feedforward=forward_expansion, dropout=dropout)
        self.transformer = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(embed_size, vocab_size)
        self.max_len = max_len

    def forward(self, tgt, memory):
        seq_len = tgt.shape[1]
        positions = torch.arange(0, seq_len).unsqueeze(0).to(tgt.device)
        tgt = self.word_embedding(tgt) + self.pos_embedding(positions)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)
        out = self.transformer(tgt.permute(1,0,2), memory.permute(1,0,2), tgt_mask=tgt_mask)
        out = self.fc(out.permute(1,0,2))
        return out

# -----------------------
# Load checkpoint
# -----------------------
MODEL_PATH = "flickr8k_model.pth"
checkpoint = torch.load(MODEL_PATH, map_location=device)
vocab = Vocabulary(checkpoint['vocab'], {idx:word for word, idx in checkpoint['vocab'].items()})

embed_size = 512
enc = EncoderCNN(embed_size).to(device)
dec = TransformerDecoder(embed_size, len(vocab.word2idx)).to(device)
enc.load_state_dict(checkpoint['encoder'])
dec.load_state_dict(checkpoint['decoder'])
enc.eval()
dec.eval()

# -----------------------
# Transforms
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# -----------------------
# Caption generation
# -----------------------
def generate_caption(image, max_len=50):
    with torch.no_grad():
        image = transform(image).unsqueeze(0).to(device)
        features = enc(image)
        tgt = torch.tensor([[vocab.word2idx["<SOS>"]]], device=device)
        caption = []
        for _ in range(max_len):
            out = dec(tgt, features)
            pred = out[:, -1, :].argmax(-1)
            if pred.item() == vocab.word2idx["<EOS>"]:
                break
            caption.append(pred.item())
            tgt = torch.cat([tgt, pred.unsqueeze(1)], dim=1)
    return " ".join([vocab.idx2word[idx] for idx in caption])

# -----------------------
# Streamlit interface
# -----------------------
st.title("Image & Video Captioning")

choice = st.radio("Choose input type:", ["Image", "Video"])

if choice == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg','jpeg','png'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Generate Caption"):
            caption = generate_caption(image)
            st.success(f"Caption: {caption}")

elif choice == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=['mp4','mov','avi'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = max(frame_count // 10, 1)  # take 10 frames roughly

        captions = []
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if i % sample_frames == 0:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                captions.append(generate_caption(image))
        cap.release()
        st.success("Video captions (sampled frames):")
        for i, c in enumerate(captions):
            st.write(f"Frame {i+1}: {c}")
