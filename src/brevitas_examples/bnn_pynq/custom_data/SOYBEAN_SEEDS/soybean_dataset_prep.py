import os
import csv
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Configuração dos diretórios e labels
base_dir = ''
classes = [
    'Broken soybeans',
    'Intact soybeans',
    'Spotted soybeans',
    'Immature soybeans',
    'Skin-damaged soybeans'
]

output_image_csv = 'images.csv'
output_label_csv = 'labels.csv'

def process_images(base_dir, classes, output_image_csv, output_label_csv):
    images = []
    labels = []

    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(base_dir, class_name)

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)

            try:
                # Carregar imagem
                with Image.open(img_path) as img:
                    img = img.resize((64,64)).convert('RGB')
                    img_array = np.array(img)  # Mantém formato H x W x C

                images.append(img_array)
                labels.append(idx)  # Usar índice da classe como rótulo
            except Exception as e:
                print(f"Erro ao processar {img_path}: {e}")

    # Salvar os dados em formato binário para preservar o formato
    np.save(output_image_csv.replace('.csv', '.npy'), images)
    np.save(output_label_csv.replace('.csv', '.npy'), labels)

    print("Processamento concluído e dados salvos como arrays NumPy.")

def save_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def split_dataset(image_file, label_file, train_split=0.85):
    # Carregar os dados
    images = np.load(image_file.replace('.csv', '.npy'))
    labels = np.load(label_file.replace('.csv', '.npy'))

    # Combinar e embaralhar
    dataset = list(zip(images, labels))
    random.shuffle(dataset)

    train_size = int(len(dataset) * train_split)
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]

    return train_data, test_data

class SoybeanDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        # Convertendo para tensor no formato C x H x W
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0  # Normalização
        label = torch.tensor(label, dtype=torch.long)  # Rótulo como escalar
        return image, label

def create_dataloaders(train_data, test_data, batch_size=32):
    train_dataset = SoybeanDataset(train_data)
    test_dataset = SoybeanDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    process_images(base_dir, classes, resize_dim, output_image_csv, output_label_csv)
    train_data, test_data = split_dataset(output_image_csv, output_label_csv)

    print(np.load(output_label_csv.replace('.csv', '.npy')))

    train_loader, test_loader = create_dataloaders(train_data, test_data)

    print(f"Dados de treino: {len(train_loader.dataset)} amostras")
    print(f"Dados de teste: {len(test_loader.dataset)} amostras")