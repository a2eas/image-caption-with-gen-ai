import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import kagglehub


#downloading data and setting up paths
path = kagglehub.dataset_download("adityajn105/flickr8k")
IMG_FOLDER = os.path.join(path, "Images")
CAP_PATH = os.path.join(path, "captions.txt")
device = "cuda" if torch.cuda.is_available() else "cpu"

#loading captions
captions_dict = defaultdict(list)

with open(CAP_PATH, encoding="utf8") as f:
    for line in f:
        line = line.strip()
        if not line or "," not in line:
            continue
        parts = line.split(",", 1)  
        img_name = parts[0].strip()
        caption = parts[1].strip()
        full_img_path = os.path.join(IMG_FOLDER, img_name)
        if os.path.isfile(full_img_path):
            captions_dict[img_name].append(caption)

print(f"Total images with captions: {len(captions_dict)}")

#vocab class that sets up padding token and start of sentance,end of sentence , unkown tokens
class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.freq_threshold = freq_threshold
        # making an id for each of these tokens
        self.word2idx = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.idx2word = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.word_freq = defaultdict(int)
        self.idx = 4
#using word2idx to give an id to each word
    def build_vocab(self, captions_dict):
        for caps in captions_dict.values():
            for caption in caps:
                for word in caption.lower().split():
                    self.word_freq[word] +=1
                    if self.word_freq[word]==self.freq_threshold:
                        self.word2idx[word] = self.idx
                        self.idx2word[self.idx] = word
                        self.idx +=1
    def numericalize(self, text):
        return [self.word2idx.get(word.lower(), self.word2idx["<UNK>"]) for word in text.split()]

vocab = Vocabulary()
vocab.build_vocab(captions_dict)
print(f"Vocabulary size: {len(vocab.word2idx)}")

#calling a function from pytorch to resize the image and turning into from numpy to tensor
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

class FlickrDataset(Dataset):
    def __init__(self, img_folder, captions_dict, vocab, transform=None):
        self.img_folder = img_folder
        self.captions_dict = captions_dict
        self.vocab = vocab
        self.transform = transform
        self.items = []
        for img_name, caps in captions_dict.items():
            for cap in caps:
                self.items.append((img_name, cap))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_name, caption = self.items[idx]
        path = os.path.join(self.img_folder, img_name)
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # putting the start and end tokens to each vector
        numericalized = [self.vocab.word2idx["<SOS>"]] + self.vocab.numericalize(caption) + [self.vocab.word2idx["<EOS>"]]
        return image, torch.tensor(numericalized)

def collate_fn(batch):
    # unzinping images and captions
    imgs, caps = zip(*batch)
    #stacking images into one tensor
    imgs = torch.stack(imgs)
    #getting length of the captions
    lengths = [len(c) for c in caps]
    #seeing maxs lenght
    max_len = max(lengths)
    #padding the captions to match the longest captions
    padded = torch.zeros(len(caps), max_len).long()
    for i, cap in enumerate(caps):
        padded[i, :lengths[i]] = cap
    return imgs, padded
#using the fucntion and then passing the return to torch dataloader
dataset = FlickrDataset(IMG_FOLDER, captions_dict, vocab, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
print(f"Dataset length: {len(dataset)}")

#the endcoder cnn this is a pre trained model is resnet18 it is already good but A++ will fine tune it to out data set :)
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in self.cnn.parameters():
            param.requires_grad = True
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
    # foward function to gett output of cnn encoder
    def forward(self, images):
        features = self.cnn(images)
        features = self.bn(features)
        return features.unsqueeze(1)  # [B, 1, embed_size] adding a dim for lentgh

class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads=8, num_layers=3, forward_expansion=2048, dropout=0.1, max_len=50):
        super().__init__()
        self.embed_size = embed_size
        #embedding our vocablary making it the same as embed size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        #getting postional embedding
        self.pos_embedding = nn.Embedding(max_len, embed_size)
        # calling pytorch transformer layer
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads,
                                                   dim_feedforward=forward_expansion, dropout=dropout)
        #stacking transformers layer
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        #ff liner layer for output
        self.fc = nn.Linear(embed_size, vocab_size)
        self.max_len = max_len
    #foward function and casual mask
    def forward(self, tgt, memory):
        seq_len = tgt.shape[1]
        positions = torch.arange(0, seq_len).unsqueeze(0).to(tgt.device)
        tgt = self.word_embedding(tgt) + self.pos_embedding(positions)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)
        out = self.transformer(tgt.permute(1,0,2), memory.permute(1,0,2), tgt_mask=tgt_mask)
        out = self.fc(out.permute(1,0,2))
        return out

#setting embedding size
embed_size = 512
# calling encoder calss and putting it onto the gpu
enc = EncoderCNN(embed_size).to(device)
#same here for the decoder
dec = TransformerDecoder(embed_size, len(vocab.word2idx)).to(device)
# loss function telling it to ignore pad tokens which are just 0 that we make the target always the same lentght
criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<PAD>"])
#optimizer algo and learning rate hyperpramter
optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-4)

def train(enc, dec, loader, device, epochs=20):
    # making sure that the encoder decoder is read to train
    enc.train()
    dec.train()
    for epoch in range(epochs):
        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        total_loss = 0
        for imgs, caps in loop:
            imgs, caps = imgs.to(device), caps.to(device)
            optimizer.zero_grad()
            features = enc(imgs)
            outputs = dec(caps[:,:-1], features)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), caps[:,1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/len(loader):.4f}")
    torch.save({
        'encoder': enc.state_dict(),
        'decoder': dec.state_dict(),
        'vocab': vocab.word2idx
    }, "flickr8k_model.pth")
    print("Model saved!")


if __name__ == "__main__":
    train(enc, dec, loader, device)
    # Example usage:
    # print(generate_caption(os.path.join(IMG_FOLDER, "2603792708_18a97bac97.jpg"),
    #                        enc, dec, vocab, transform))

