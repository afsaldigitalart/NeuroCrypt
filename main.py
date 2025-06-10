import torch

def text_to_bit(text, chunk_size=120):
    bits = "".join(format(ord(char), "08b") for char in text)
    if len(bits) % chunk_size != 0:
        padding =  chunk_size - len(bits) % chunk_size
        bits = bits.ljust(len(bits)+padding, '0')

    chunks = [bits[i:i+chunk_size] for i in range(0, len(bits), chunk_size)]
    return [torch.tensor([int(b) for b in chunk], dtype=torch.float32) for chunk in chunks]

def bit_to_text(chunks):
    all_bits = ''.join(''.join(str(int(b.item())) for b in chunk) for chunk in chunks)
    chars = [chr(int(all_bits[i:i+8], 2)) for i in range(0, len(all_bits), 8)]
    return ''.join(chars).rstrip('\x00')

with open("info.txt", "r") as file: 
    x = text_to_bit(file.read())
y = bit_to_text(x)

print(x)
print("------------------------------------")
print(y)