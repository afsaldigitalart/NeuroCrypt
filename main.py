import torch
import torch.nn as nn

def text_to_bit(text, chunk_size=32):
    text_ascii = text.encode('ascii', errors='replace')
    length_prefix = len(text_ascii).to_bytes(2, byteorder='big')
    all_bytes = length_prefix + text_ascii
    bits = ''.join(format(byte, '08b') for byte in all_bytes)
    if len(bits) % chunk_size != 0:
        bits = bits.ljust(len(bits) + (chunk_size - len(bits) % chunk_size), '0')
    chunks = [bits[i:i+chunk_size] for i in range(0, len(bits), chunk_size)]
    return [torch.tensor([int(b) for b in chunk], dtype=torch.float32) for chunk in chunks]

def bit_to_text(chunks):
    all_bits = ''.join(''.join('1' if b.item() > 0.5 else '0' for b in chunk) for chunk in chunks)
    text_length = int(all_bits[:16], 2)
    required_bits = text_length * 8
    text_bits = all_bits[16:16+required_bits]
    bytes_data = [int(text_bits[i:i+8], 2) for i in range(0, len(text_bits), 8)]
    return ''.join(chr(b) if 32 <= b <= 126 else '?' for b in bytes_data)

class Encrypter(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(size*3, 512), 
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, size)
        )

    def forward(self, msg, key, nonce):
        return self.model(torch.cat([msg, key, nonce], dim=1))

class Decrypter(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(size*3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, size)
        )
    def forward(self, cipher, key, nonce):
        return self.model(torch.cat([cipher, key, nonce], dim=1))

class Detective(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, size)
        )
    def forward(self, cipher):
        return self.model(cipher)