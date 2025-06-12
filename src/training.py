import torch
import torch.nn as nn
import torch.optim as optim
import main as m
import matplotlib.pyplot as plt
import os
import random


SIZE = 64
EPOCHS = 40000
BATCH = 128

def gradient_penalty(eve, cipher):
    cipher.requires_grad_(True)
    output = eve(cipher)
    grad = torch.autograd.grad(outputs=output, inputs=cipher,
                                grad_outputs=torch.ones_like(output),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = grad.pow(2).mean()
    return gp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

encrypter = m.Encrypter(SIZE).to(device)
decrypter = m.Decrypter(SIZE).to(device)
detective = m.Detective(SIZE).to(device)

opt_ab = optim.Adam(list(encrypter.parameters()) + list(decrypter.parameters()), lr=2e-4)
opt_eve = optim.Adam(detective.parameters(), lr=2e-5)

criterion = nn.BCEWithLogitsLoss()
lambda_ab_loss = 1.8
lambda_gp = 10.0

decrypter_loss = []
detect_loss = []
decrypter_accuracies = []
detective_accuracies = []

for epoch in range(1, EPOCHS+1):

    msg = torch.randint(0, 2, (BATCH, SIZE)).float().to(device)
    key = torch.randint(0, 2, (BATCH, SIZE)).float().to(device)
    nonce = torch.randint(0, 2, (BATCH, SIZE)).float().to(device)

    cipher_msg = encrypter(msg, key, nonce)
    cipher_noisy = cipher_msg + torch.randn_like(cipher_msg)*(0.03 + 0.01 * random.random())
    decrypt_out = decrypter(cipher_noisy, key, nonce)
    detective_out = detective(cipher_noisy.detach())

    decrypt_loss = criterion(decrypt_out, msg)
    detective_loss = criterion(detective_out, msg)
    gp_loss = gradient_penalty(detective, cipher_noisy.detach())
    ab_loss = decrypt_loss + lambda_ab_loss * (1.0 - detective_loss)

    opt_ab.zero_grad()
    ab_loss.backward(retain_graph=True)
    opt_ab.step()
    
    opt_eve.zero_grad()
    (detective_loss + lambda_gp * gp_loss).backward()
    opt_eve.step()

    bob_acc = ((torch.sigmoid(decrypt_out) > 0.45) == msg).float().mean().item()
    eve_acc = ((torch.sigmoid(detective_out) > 0.45) == msg).float().mean().item()

    decrypter_loss.append(decrypt_loss.item())
    detect_loss.append(detective_loss.item())
    decrypter_accuracies.append(bob_acc)
    detective_accuracies.append(eve_acc)

    if epoch % 200 == 0 or epoch == 1:
        print(f"Epoch {epoch} | Bob Acc: {bob_acc*100:.2f}% | Eve Acc: {eve_acc*100:.2f}%")

os.makedirs("models", exist_ok=True)
torch.save(encrypter.state_dict(), "models/encrypter.pth")
torch.save(decrypter.state_dict(), "models/decrypter.pth")
torch.save(detective.state_dict(), "models/detective.pth")
print("Training complete!")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(decrypter_loss, label="Bob Loss")
plt.plot(detect_loss, label="Eve Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(decrypter_accuracies, label="Bob Accuracy")
plt.plot(detective_accuracies, label="Eve Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("training_graph.png")