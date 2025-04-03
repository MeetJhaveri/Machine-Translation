import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import random
import os
import urllib.request
import gzip
import shutil
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import translate_sentence, save_checkpoint, load_checkpoint, bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if not os.path.exists("data"):
    os.makedirs("data")

def download_and_extract(url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {url}...")
        gz_path = output_path + ".gz"
        urllib.request.urlretrieve(url, gz_path)
        
        print(f"Extracting to {output_path}...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        os.remove(gz_path)  
    else:
        print(f"File {output_path} already exists, skipping download.")


TRAIN_DE_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.de.gz"
TRAIN_EN_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.en.gz"
VAL_DE_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.de.gz"
VAL_EN_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.en.gz"
TEST_DE_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.de.gz"
TEST_EN_URL = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.en.gz"


download_and_extract(TRAIN_DE_URL, "data/train.de")
download_and_extract(TRAIN_EN_URL, "data/train.en")
download_and_extract(VAL_DE_URL, "data/val.de")
download_and_extract(VAL_EN_URL, "data/val.en")
download_and_extract(TEST_DE_URL, "data/test.de")
download_and_extract(TEST_EN_URL, "data/test.en")

try:
    spacy_ger = spacy.load("de_core_news_sm")
    spacy_eng = spacy.load("en_core_web_sm")
except OSError:
    print("Please install the required spaCy models:")
    print("python -m spacy download de_core_news_sm")
    print("python -m spacy download en_core_web_sm")
    exit(1)

def tokenize_ger(text):
    return [token.text.lower() for token in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    return [token.text.lower() for token in spacy_eng.tokenizer(text)]

def read_data(de_path, en_path):
    data = []
    with open(de_path, 'r', encoding='utf-8') as de_file, open(en_path, 'r', encoding='utf-8') as en_file:
        for de_line, en_line in zip(de_file, en_file):
            data.append((de_line.strip(), en_line.strip()))
    return data

train_data = read_data("data/train.de", "data/train.en")
val_data = read_data("data/val.de", "data/val.en")
test_data = read_data("data/test.de", "data/test.en")

print(f"Train data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")
print(f"Test data size: {len(test_data)}")

class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.itos = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        
    def build_vocabulary(self, sentences, tokenizer, min_freq=2):
        word_counts = {}
        for sentence in sentences:
            for word in tokenizer(sentence):
                word_counts[word] = word_counts.get(word, 0) + 1
        idx = 4
        for word, count in sorted(word_counts.items(), key=lambda x: -x[1]):
            if count >= min_freq:
                self.itos.append(word)
                self.stoi[word] = idx
                idx += 1
        print(f"Built {self.name} vocabulary with {len(self.itos)} words")
        
    def __len__(self):
        return len(self.itos)
    
    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi["<unk>"])
    
    def lookup_tokens(self, indices):
        return [self.itos[idx] for idx in indices]

german_vocab = Vocabulary("German")
english_vocab = Vocabulary("English")
german_vocab.build_vocabulary([pair[0] for pair in train_data], tokenize_ger, min_freq=2)
english_vocab.build_vocabulary([pair[1] for pair in train_data], tokenize_eng, min_freq=2)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(
            embedding_size, 
            hidden_size, 
            num_layers, 
            dropout=dropout, 
            bidirectional=True,
            batch_first=False
        )
        
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedding = self.dropout(self.embedding(x))
        
        encoder_states, (hidden, cell) = self.rnn(embedding)
        
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))
        
        return encoder_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(
            hidden_size * 2 + embedding_size,
            hidden_size, 
            num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False
        )
        
        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        
    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)
        
        embedding = self.dropout(self.embedding(x))
        
        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        
        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        
        attention = self.softmax(energy)
        
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)
        
        rnn_input = torch.cat((context_vector, embedding), dim=2)
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        predictions = self.fc(output).squeeze(0)
        
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        encoder_states, hidden, cell = self.encoder(source)
        
        x = target[0]
        
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            
            outputs[t] = output
            
            best_guess = output.argmax(1)
            
            x = target[t] if random.random() < teacher_force_ratio else best_guess
            
        return outputs

def preprocess_data(batch):
    src_batch, trg_batch = [], []
    for src_text, trg_text in batch:
        src_tokens = [german_vocab["<sos>"]] + [german_vocab[token] for token in tokenize_ger(src_text)] + [german_vocab["<eos>"]]
        trg_tokens = [english_vocab["<sos>"]] + [english_vocab[token] for token in tokenize_eng(trg_text)] + [english_vocab["<eos>"]]
        src_batch.append(torch.tensor(src_tokens, dtype=torch.long))
        trg_batch.append(torch.tensor(trg_tokens, dtype=torch.long))
    src_batch = pad_sequence(src_batch, padding_value=german_vocab["<pad>"])
    trg_batch = pad_sequence(trg_batch, padding_value=english_vocab["<pad>"])
    return src_batch, trg_batch

input_size_encoder = len(german_vocab)
input_size_decoder = len(english_vocab)
output_size = len(english_vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 1
enc_dropout = 0.0
dec_dropout = 0.0
learning_rate = 3e-4
batch_size = 32
num_epochs = 20
load_model = False
save_model = True

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn=preprocess_data)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, collate_fn=preprocess_data)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, collate_fn=preprocess_data)

encoder_net = Encoder(
    input_size_encoder, 
    encoder_embedding_size, 
    hidden_size, 
    num_layers, 
    enc_dropout
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout
).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pad_idx = english_vocab["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

writer = SummaryWriter("runs/translation")

if load_model:
    load_checkpoint(torch.load("checkpoint.pth.tar"), model, optimizer)

test_sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

step = 0
for epoch in range(num_epochs):
    print(f"\n[Epoch {epoch+1}/{num_epochs}]")
    
    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
    
    model.eval()
    translated_sentence = translate_sentence(
        model, test_sentence, german_vocab, english_vocab, device, max_length=50
    )
    print(f"Translated example sentence: \n {' '.join(translated_sentence)}")
    
    model.train()
    
    for batch_idx, (src, trg) in enumerate(train_loader):
        src, trg = src.to(device), trg.to(device)
        
        output = model(src, trg)
        
        output = output[1:].reshape(-1, output.shape[2])
        trg = trg[1:].reshape(-1)
        
        optimizer.zero_grad()
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        
        optimizer.step()
        
        writer.add_scalar("Training loss", loss.item(), global_step=step)
        step += 1
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

test_subset = test_data[:100]  
print("\nCalculating BLEU score...")
score = bleu(test_subset, model, german_vocab, english_vocab, device)
print(f"BLEU score: {score * 100:.2f}")

print("=> Saving trained model")
torch.save(model.state_dict(), "trained_model.pth")
print("Training complete!")