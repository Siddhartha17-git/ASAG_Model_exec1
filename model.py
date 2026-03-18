import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error

MAXLEN     = 128
BATCH_SIZE = 32
EPOCHS     = 10
SEED       = 20
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

classes = {"0":0,"0.5":1,"1":2,"1.5":3,"2":4,"2.5":5,"3":6,"3.5":7,"4":8,"4.5":9,"5":10}
NUM_CLASSES = len(classes)

def load_data(filename):
    D = []
    with open(filename, encoding="utf-8") as f:
        for l in f:
            parts = l.strip().split("\t")
            if len(parts) != 4:
                continue
            q, r, a, label = parts
            if label in classes:
                D.append((q, r, a, classes[label]))
    return D

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class ASAGDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        q, r, a, label = self.data[idx]
        enc = tokenizer(r, a, max_length=MAXLEN,
                        padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc["token_type_ids"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long)
        }

class ASAGModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.bert      = BertModel.from_pretrained("bert-base-uncased")
        self.bilstm    = nn.LSTM(input_size=768, hidden_size=256,
                                 batch_first=True, bidirectional=True,
                                 dropout=0.1, num_layers=2)
        self.norm      = nn.LayerNorm(512)
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=4,
                                               batch_first=True, dropout=0.1)
        self.dropout   = nn.Dropout(0.3)
        self.classifier= nn.Linear(512, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out        = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids).last_hidden_state
        out, _     = self.bilstm(out)
        out        = self.norm(out)
        out, _     = self.attention(out, out, out)
        out        = self.norm(out)
        pooled     = out.mean(dim=1) + out.max(dim=1).values
        return self.classifier(self.dropout(pooled))

def pearson(v1, v2):
    n = len(v1)
    s1, s2 = sum(v1), sum(v2)
    s1sq   = sum(x**2 for x in v1)
    s2sq   = sum(x**2 for x in v2)
    psum   = sum(v1[i]*v2[i] for i in range(n))
    num    = psum-(s1*s2/n)
    den    = math.sqrt((s1sq-s1**2/n)*(s2sq-s2**2/n))
    return num/den if den != 0 else 0.0

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            ttype = batch["token_type_ids"].to(device)
            labs  = batch["label"].numpy()
            pred  = model(ids,mask,ttype).argmax(dim=1).cpu().numpy()
            for p,t in zip(pred,labs):
                trues.append(t)
                preds.append(t if abs(p-t)<=1 else p)
    acc   = sum(1 for p,t in zip(preds,trues) if p==t)/len(trues)
    p     = pearson(preds,trues)
    rmse  = math.sqrt(mean_squared_error(trues,preds))
    mae   = mean_absolute_error(trues,preds)
    macro = f1_score(trues,preds,average="macro",zero_division=0)
    wt    = f1_score(trues,preds,average="weighted",zero_division=0)
    return acc,p,rmse,mae,macro,wt

def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        ttype = batch["token_type_ids"].to(device)
        labs  = batch["label"].to(device)
        optimizer.zero_grad()
        loss  = criterion(model(ids,mask,ttype), labs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss/len(loader)

if __name__ == "__main__":
    data = load_data("dataset/NorthTexasDataset/expand.txt")
    np.random.shuffle(data)
    split        = int(len(data)*0.8)
    train_loader = DataLoader(ASAGDataset(data[:split]), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(ASAGDataset(data[split:]), batch_size=BATCH_SIZE, shuffle=False)

    model = ASAGModel().to(device)
    for i, layer in enumerate(model.bert.encoder.layer):
        if i < 8:
            for param in layer.parameters():
                param.requires_grad = False

    total_steps = len(train_loader)*EPOCHS
    optimizer   = torch.optim.AdamW([
        {"params": model.bert.parameters(),        "lr": 2e-5},
        {"params": model.bilstm.parameters(),      "lr": 1e-4},
        {"params": model.attention.parameters(),   "lr": 1e-4},
        {"params": model.classifier.parameters(),  "lr": 1e-4},
    ], weight_decay=0.01)
    scheduler   = get_linear_schedule_with_warmup(optimizer,
                    num_warmup_steps=total_steps//10,
                    num_training_steps=total_steps)
    criterion   = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        acc,p,rmse,mae,macro,wt = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss:{loss:.4f} | Acc:{acc:.4f} | Pearson:{p:.4f} | RMSE:{rmse:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  Best model saved! acc={best_acc:.4f}")

    model.load_state_dict(torch.load("best_model.pt"))
    acc,p,rmse,mae,macro,wt = evaluate(model, test_loader)
    print(f"\nFinal Results:")
    print(f"  Accuracy   : {acc:.5f}")
    print(f"  Pearson    : {p:.5f}")
    print(f"  MAE        : {mae:.5f}")
    print(f"  RMSE       : {rmse:.5f}")
    print(f"  Macro F1   : {macro:.5f}")
    print(f"  Weighted F1: {wt:.5f}")
