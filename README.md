# image-caption-with-gen-ai

A **PyTorch + Streamlit project** that generates captions for images and videos using a **CNN encoder + Transformer decoder** trained on the **Flickr8k dataset**.

---

## Features

* **Image captioning**: Upload an image and get a descriptive caption.
* **Video captioning**: Upload a video; captions are generated for sampled frames.
* **Transformer-based decoder**: Auto-regressive generation with causal masking.
* **CNN encoder**: Pretrained ResNet18 for image feature extraction.
* **Streamlit interface**: Simple web UI for users to upload media and get captions.
* **Model saving/loading**: Trained model can be saved and reused.

---

## Folder Structure

```
project_folder/
│
├── flickr8k_model.pth       # Trained model weights
├── streamlit_app.py          # Streamlit interface
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
```

---

## Installation

1. **Clone the repository** (or copy files into a folder):

```bash
git clone https://github.com/a2eas/image-caption-with-gen-ai
cd project_folder
```

2. **Create a virtual environment** (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

`requirements.txt` example:

```
torch
torchvision
Pillow
opencv-python
tqdm
streamlit
```

---

## Usage

### Run Streamlit app

```bash
streamlit run streamlit_app.py
```

* A browser window will open the interface.
* Select **Image** or **Video**.
* Upload a file and click **Generate Caption**.
* For videos, captions are generated for sampled frames (~10 frames per video).

---

## Model

* **Encoder**: ResNet18 (pretrained), fine-tuned on Flickr8k images.

* **Decoder**: Transformer decoder with:

  * Positional embeddings
  * Causal mask to prevent looking ahead
  * Linear layer projecting embeddings to vocabulary size

* **Vocabulary**: Built from Flickr8k captions; includes `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>` tokens.

---

## Training

If you want to retrain the model:

```python
# In Python
from train_script import train, EncoderCNN, TransformerDecoder, FlickrDataset

train(enc, dec, loader, device, epochs=20)
```

* Uses **CrossEntropyLoss** with padding ignored.
* Optimizer: Adam, learning rate 1e-4.

---

## Notes

* GPU recommended for training and inference.
* For small datasets like Flickr8k (~8k images), training the full CNN may overfit. Consider freezing most CNN layers and fine-tuning only the last layer.
* Model saved as `flickr8k_model.pth` includes:

  * Encoder weights
  * Decoder weights
  * Vocabulary (`word2idx`)


## License

MIT License. Free for academic and personal use.

