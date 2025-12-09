import streamlit as st
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import json

# 1. SETUP & CONFIGURATION
st.set_page_config(page_title="Name Origin Predictor", layout="wide")
DEVICE = torch.device('cpu') # Use CPU for free cloud tiers (GPUs cost money)

# Define paths relative to this file
BASE_DIR = Path(__file__).parent.resolve()

# Update these paths to match where you put files in your 'models' folder
MODEL_REGISTRY = {
    'mbert_base': {
        'type': 'mbert',
        'path': BASE_DIR / 'models' / 'mbert_model', 
        'label': 'mBERT'
    },
    'lstm_orig': {
        'type': 'lstm',
        'path': BASE_DIR / 'models' / 'lstm_runs' / 'lstm_best.pt',
        'label': 'LSTM'
    },
    # Add other models here if you uploaded them
}

# 2. MODEL DEFINITIONS (From your notebook)
class SimpleLSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dropout = cfg.get('DROPOUT', 0.0)
        self.embedding = nn.Embedding(cfg['vocab_size'], cfg['EMBED_DIM'], padding_idx=cfg['PAD_IDX'])
        self.lstm = nn.LSTM(
            cfg['EMBED_DIM'], cfg['HIDDEN_SIZE'], num_layers=cfg['NUM_LAYERS'],
            batch_first=True, bidirectional=cfg['BIDIRECTIONAL'],
            dropout=dropout if cfg['NUM_LAYERS'] > 1 else 0.0,
        )
        out_dim = cfg['HIDDEN_SIZE'] * (2 if cfg['BIDIRECTIONAL'] else 1)
        self.fc = nn.Linear(out_dim, cfg['num_classes'])

    def forward(self, x, lengths=None):
        emb = self.embedding(x)
        if lengths is not None:
            packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(emb)
        feat = torch.cat((hidden[-2], hidden[-1]), dim=1) if self.lstm.bidirectional else hidden[-1]
        return self.fc(feat)

class DeepLSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dropout = cfg.get('DROPOUT', 0.0)
        self.embedding = nn.Embedding(cfg['vocab_size'], cfg['EMBED_DIM'], padding_idx=cfg['PAD_IDX'])
        self.lstm = nn.LSTM(
            cfg['EMBED_DIM'], cfg['HIDDEN_SIZE'], num_layers=cfg['NUM_LAYERS'],
            batch_first=True, bidirectional=cfg['BIDIRECTIONAL'],
            dropout=dropout if cfg['NUM_LAYERS'] > 1 else 0.0,
        )
        out_dim = cfg['HIDDEN_SIZE'] * (2 if cfg['BIDIRECTIONAL'] else 1)
        hid = cfg.get('CLASSIFIER_HIDDEN')
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, cfg['num_classes']),
        )

    def forward(self, x, lengths=None):
        emb = self.embedding(x)
        if lengths is not None:
            packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(emb)
        feat = torch.cat((hidden[-2], hidden[-1]), dim=1) if self.lstm.bidirectional else hidden[-1]
        return self.classifier(feat)

# 3. HELPER FUNCTIONS
def encode_names(names, stoi, max_len, pad_idx, unk_idx, device):
    encoded, lengths = [], []
    for name in names:
        chars = list(str(name))[:max_len]
        ids = [stoi.get(ch, unk_idx) for ch in chars]
        if not ids: ids = [unk_idx]
        length = min(len(ids), max_len)
        if len(ids) < max_len:
            ids += [pad_idx] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        encoded.append(ids)
        lengths.append(length)
    x = torch.tensor(encoded, dtype=torch.long, device=device)
    lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    return x, lengths

def _normalize_idx2label(raw_idx2label, label2idx):
    if raw_idx2label: return {int(k): v for k, v in raw_idx2label.items()}
    return {v: k for k, v in label2idx.items()}

# 4. LOADERS (Cached for performance)
@st.cache_resource
def load_model(model_key):
    spec = MODEL_REGISTRY[model_key]
    if spec['type'] == 'mbert':
        model_dir = spec['path']
        with open(model_dir / 'meta.json', 'r') as f: meta = json.load(f)
        label2idx = meta['label2idx']
        idx2label = _normalize_idx2label(meta.get('idx2label'), label2idx)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        model.to(DEVICE).eval()
        return {'type': 'mbert', 'model': model, 'tokenizer': tokenizer, 'idx2label': idx2label, 'max_len': meta.get('max_len', 40)}
    else:
        ckpt = torch.load(spec['path'], map_location=DEVICE)
        cfg = ckpt['config']
        label2idx = ckpt['label2idx']
        idx2label = {int(k): v for k, v in ckpt['idx2label'].items()}
        if 'fc.weight' in ckpt['state_dict']: model = SimpleLSTM(cfg)
        else: model = DeepLSTM(cfg)
        model.load_state_dict(ckpt['state_dict'])
        model.to(DEVICE).eval()
        return {'type': 'lstm', 'model': model, 'stoi': ckpt['stoi'], 'idx2label': idx2label, 'max_len': cfg['MAX_LEN'], 'pad_idx': cfg.get('PAD_IDX', 0), 'unk_idx': cfg.get('UNK_IDX', 1)}

# 5. UI LAYOUT
st.title("🌍 Name-to-Country Predictor")
st.markdown("Enter names below to predict their likely country of origin.")

col1, col2 = st.columns([1, 2])

with col1:
    model_choice = st.selectbox("Choose Model", list(MODEL_REGISTRY.keys()), format_func=lambda x: MODEL_REGISTRY[x]['label'])
    top_k = st.slider("Top K Predictions", 1, 5, 3)
    
names_input = st.text_area("Enter Names (one per line)", "Aston Escalera\nWei Zhang\nFatima Ali", height=150)
predict_btn = st.button("Predict", type="primary")

if predict_btn and names_input:
    names = [n.strip() for n in names_input.splitlines() if n.strip()]
    
    with st.spinner(f"Loading {MODEL_REGISTRY[model_choice]['label']}..."):
        bundle = load_model(model_choice)
        
    with st.spinner("Predicting..."):
        results = []
        with torch.no_grad():
            if bundle['type'] == 'mbert':
                enc = bundle['tokenizer'](names, padding=True, truncation=True, max_length=bundle['max_len'], return_tensors='pt')
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                logits = bundle['model'](**enc).logits
            else:
                x, lengths = encode_names(names, bundle['stoi'], bundle['max_len'], bundle['pad_idx'], bundle['unk_idx'], DEVICE)
                logits = bundle['model'](x, lengths)
                
            probs = logits.softmax(dim=-1).cpu()
            scores, idxs = probs.topk(min(top_k, probs.shape[1]), dim=-1)
            
            for name, score_row, idx_row in zip(names, scores, idxs):
                top_preds = []
                for s, i in zip(score_row, idx_row):
                    top_preds.append((bundle['idx2label'][i.item()], float(s.item())))
                results.append({"Name": name, "Predictions": top_preds})

    # Display Results
    for res in results:
        with st.expander(f"**{res['Name']}** - Top prediction: {res['Predictions'][0][0].title()}", expanded=True):
            df = pd.DataFrame(res['Predictions'], columns=["Country", "Confidence"])
            df['Confidence'] = df['Confidence'].map('{:.1%}'.format)
            st.table(df)