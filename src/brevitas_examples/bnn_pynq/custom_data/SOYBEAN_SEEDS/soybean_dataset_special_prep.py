import os
import csv
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Configurações das classes
classes = ['Broken soybeans', 'Intact soybeans', 'Spotted soybeans', 'Immature soybeans', 'Skin-damaged soybeans']
output_image_csv = 'images.csv'
output_label_csv = 'labels.csv'

# Transformação avançada
enhanced_transform = transforms.Compose([
    transforms.RandomRotation(45),                     # Rotação aleatória até 20 graus
    transforms.RandomHorizontalFlip(),           # Espelhamento horizontal com 50% de probabilidade
    transforms.RandomVerticalFlip(),             # Espelhamento vertical com 20% de probabilidade
    transforms.RandomCrop(32, padding=4),         # Corte aleatório de 32x32 com padding de 4
    transforms.ToTensor(),                            # Conversão para tensor
])

def process_images(base_dir, classes, output_image_csv, output_label_csv):
    images, labels = [], []
    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(base_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    img = img.resize((32, 32)).convert('RGB')
                    img_array = np.array(img)
                images.append(img_array)
                labels.append(idx)
            except Exception as e:
                print(f"Erro ao processar {img_path}: {e}")
    np.save(output_image_csv.replace('.csv', '.npy'), images)
    np.save(output_label_csv.replace('.csv', '.npy'), labels)
    print("Processamento concluído e dados salvos como arrays NumPy.")

def augment_images(images, labels, target_class, augment_count, transform=enhanced_transform):
    augmented_images, augmented_labels = [], []
    class_images = [img for img, lbl in zip(images, labels) if lbl == target_class]
    while len(augmented_images) < augment_count:
        img = random.choice(class_images)
        img = Image.fromarray(img)
        img_augmented = transform(img).numpy().transpose(1, 2, 0) * 255  # Volta para H x W x C
        augmented_images.append(img_augmented.astype(np.uint8))
        augmented_labels.append(target_class)
    return augmented_images, augmented_labels

def create_custom_split(image_file, label_file):
    images = np.load(image_file.replace('.csv', '.npy'))
    labels = np.load(label_file.replace('.csv', '.npy'))

    # Separar classes
    new_labels = []
    for lbl in labels:
        if lbl in [0, 2, 4]:  # Abnormal: Broken, Spotted, Skin-damaged
            new_labels.append(2)
        elif lbl == 3:  # Immature
            new_labels.append(0)
        else:  # Intact
            new_labels.append(1)

    # Balanceamento com data augmentation
    images, labels = list(images), list(new_labels)
    num_to_add = 2000

    for target_class in [0, 1]:  # Immature e Intact
        aug_images, aug_labels = augment_images(images, labels, target_class, num_to_add)
        images.extend(aug_images)
        labels.extend(aug_labels)

    # Embaralhar e combinar
    dataset = list(zip(images, labels))
    random.shuffle(dataset)
    train_size = int(len(dataset) * 0.8)
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]
    return train_data, test_data

class SoybeanDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = Image.fromarray(image)  # H x W x C
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

def create_dataloaders(train_data, test_data, batch_size=32):
    train_dataset = SoybeanDataset(train_data, transform=enhanced_transform)
    test_dataset = SoybeanDataset(test_data, transform=enhanced_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Função para salvar 100 imagens aleatórias
def save_random_images(dataloader, output_dir, num_images=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_images = 0
    dataloader = iter(dataloader)  # Torna o dataloader iterável

    while saved_images < num_images:
        batch = next(dataloader, None)
        if batch is None:
            break  # Se não houver mais dados, saia do loop
        images, labels = batch
        for i, image in enumerate(images):
            if saved_images >= num_images:
                break
            image = image.permute(1, 2, 0).numpy() * 255  # Converte para HxWxC
            image = Image.fromarray(image.astype('uint8'))
            image_name = f"{saved_images + 1}.png"
            image.save(os.path.join(output_dir, image_name))
            saved_images += 1

if __name__ == "__main__":
    # Exemplo de uso
    base_dir = './'
    process_images(base_dir, classes, output_image_csv, output_label_csv)
    train_data, test_data = create_custom_split(output_image_csv, output_label_csv)

    train_loader, test_loader = create_dataloaders(train_data, test_data)

    output_dir = './random_images'
    save_random_images(train_loader, output_dir, num_images=100)
    print("Imagens salvas com sucesso!")

    print(f"Dados de treino: {len(train_loader.dataset)} amostras")
    print(f"Dados de teste: {len(test_loader.dataset)} amostras")