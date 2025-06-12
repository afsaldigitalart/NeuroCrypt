import torch
import main as m

SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
encrypter = m.Encrypter(SIZE)
decrypter = m.Decrypter(SIZE)
detective = m.Detective(SIZE)

encrypter.load_state_dict(torch.load(r"models\encrypter.pth", map_location=device))
decrypter.load_state_dict(torch.load(r"models\decrypter.pth", map_location=device))
detective.load_state_dict(torch.load(r"models\detective.pth", map_location=device))

encrypter.eval()
decrypter.eval()
detective.eval()

text = input("Enter a Dummy Text: ")

print(f"\nTesting: {text}")
chunks = m.text_to_bit(text, SIZE)
decrypter_outs, detective_outs = [], []
key = torch.randint(0, 2, (1, SIZE)).float().to(device)
nonce = torch.randint(0, 2, (1, SIZE)).float().to(device)

for chunk in chunks:
    msg = chunk.unsqueeze(0).to(device)
    with torch.no_grad():
        cipher = encrypter(msg, key, nonce)
        decrypter_out = decrypter(cipher, key, nonce)
        detective_out = detective(cipher)

    decrypter_outs.append(torch.sigmoid(decrypter_out).squeeze(0).cpu())
    detective_outs.append(torch.sigmoid(detective_out).squeeze(0).cpu())

decrypter_decoded = m.bit_to_text(decrypter_outs)
detective_decoded = m.bit_to_text(detective_outs)

print(f"Decrypter: {decrypter_decoded}")
print(f"Detective: {detective_decoded}")

if decrypter_decoded == text:
    print("Decrypter: Perfect decryption!")
else:
    print("Decrypter: Decryption failed.")
if detective_decoded == text:
    print("Detective: SECURITY BREACH!")
elif "Error" in detective_decoded:
    print("Detective: Failed (Good Job!)")
else:
    print("Detective: Partial decoding")

