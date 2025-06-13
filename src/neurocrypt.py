import torch
import torch.nn as nn

SIZE = 64 #Size of the message/key/nonce tensors. Default is 64.

def text_to_bit(text, chunk_size=32):

    """
    Converts a given text string into a list of binary tensors.

    Args:
        text (str): The input text to encode.
        chunk_size (int): Number of bits in each tensor chunk. Default is 32.

    Returns:
        List[torch.Tensor]: A list of float tensors, each representing a chunk of binary-encoded text.
                            The first 16 bits encode the length of the original text.
    """
        
    text_ascii = text.encode('ascii', errors='replace')
    length_prefix = len(text_ascii).to_bytes(2, byteorder='big')
    all_bytes = length_prefix + text_ascii
    bits = ''.join(format(byte, '08b') for byte in all_bytes)
    if len(bits) % chunk_size != 0:
        bits = bits.ljust(len(bits) + (chunk_size - len(bits) % chunk_size), '0')
    chunks = [bits[i:i+chunk_size] for i in range(0, len(bits), chunk_size)]
    return [torch.tensor([int(b) for b in chunk], dtype=torch.float32) for chunk in chunks]

def bit_to_text(chunks):

    """
    Reconstructs the original text string from a list of binary tensor chunks.

    Args:
        chunks (List[torch.Tensor]): A list of float tensors, each containing bits (values close to 0 or 1).

    Returns:
        str: The decoded ASCII text. Non-printable characters are replaced with '?'.
    """
        
    all_bits = ''.join(''.join('1' if b.item() > 0.5 else '0' for b in chunk) for chunk in chunks)
    text_length = int(all_bits[:16], 2)
    required_bits = text_length * 8
    text_bits = all_bits[16:16+required_bits]
    bytes_data = [int(text_bits[i:i+8], 2) for i in range(0, len(text_bits), 8)]
    return ''.join(chr(b) if 32 <= b <= 126 else '?' for b in bytes_data)


class Encrypter(nn.Module):

    """
    Neural network module for simulating encryption.

    This model takes a message, key, and nonce, and produces a ciphertext tensor.
    It uses a simple feedforward architecture to model the encryption process.

    Args:
        size (int): Size of the message/key/nonce tensors. Default is 64.

    Forward Inputs:
        msg (Tensor): Message tensor of shape (batch_size, size).
        key (Tensor): Key tensor of shape (batch_size, size).
        nonce (Tensor): Nonce tensor of shape (batch_size, size).

    Forward Output:
        Tensor: Encrypted tensor (ciphertext) of shape (batch_size, size)."""

    def __init__(self, size=64):
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
        
    """
    Neural network module for simulating decryption.

    This model attempts to reconstruct the original message from the ciphertext,
    using the key and nonce as inputs.

    Args:
        size (int): Size of the ciphertext/key/nonce tensors. Default is 64.

    Forward Inputs:
        cipher (Tensor): Ciphertext tensor of shape (batch_size, size).
        key (Tensor): Key tensor of shape (batch_size, size).
        nonce (Tensor): Nonce tensor of shape (batch_size, size).

    Forward Output:
        Tensor: Reconstructed message tensor of shape (batch_size, size).
    """
        
    def __init__(self, size=64):
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
        
    """
    Adversarial neural network that attempts to decode ciphertext without knowing the key or nonce.

    This simulates an attacker trying to decrypt the message by analyzing ciphertext alone.

    Args:
        size (int): Size of the ciphertext and output tensors. Default is 64.

    Forward Input:
        cipher (Tensor): Ciphertext tensor of shape (batch_size, size).

    Forward Output:
        Tensor: Predicted message tensor of shape (batch_size, size).
    """
    
    def __init__(self, size=64):
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