import random
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
from utils import get_PIL_image

MODEL_SAVE_PATH = "models/stylist_model.pth"

transform = transforms.Compose([
        transforms.Resize((400, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

class TripletDataset(Dataset):
    def __init__(self, categories, recommends, photos_with_categories, transform=None, triplets_per_anchor=100):
        self.categories = categories
        self.recommends = recommends
        self.photos_with_categories = photos_with_categories
        self.triplets_per_anchor = triplets_per_anchor
        self.transform = transform

        all_categories = [item['category'] for item in recommends]

        categories_ = dict(categories)
        for category in categories_.keys():
            if category not in all_categories:
                del categories[category]
        
        unrelated_photos = defaultdict(list)
        for photo_url, product_category in photos_with_categories.items():
            for category, unrelated_categories in categories.items():
                if product_category in unrelated_categories:
                    unrelated_photos[category].append(photo_url)
          
        self.unrelated_photos = unrelated_photos

    def __len__(self):
        return len(self.recommends) * self.triplets_per_anchor

    def __getitem__(self, idx):
        anchor = random.choice(self.recommends)
        anchor_image = anchor["photo"]
        anchor_category = anchor["category"]

        positive_image = random.choice(anchor["recommends"])

        negative_candidates = self.unrelated_photos.get(anchor_category, [])
        negative_image = random.choice(negative_candidates)
        negative_PIL = get_PIL_image(negative_image)

        while negative_PIL is None:
            negative_image = random.choice(negative_candidates)
            negative_PIL = get_PIL_image(negative_image)
            if negative_PIL is not None:
                break

        if self.transform:
            anchor_image = self.transform(get_PIL_image(anchor_image))
            positive_image = self.transform(get_PIL_image(positive_image))
            negative_image = self.transform(negative_PIL)

        return anchor_image, positive_image, negative_image
    
class EmbeddingModel(nn.Module):
    def __init__(self):
        super(EmbeddingModel, self).__init__()
        base_model = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0)
        return loss.mean()
    
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for anchor, positive, negative in tqdm(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        anchor_embed = model(anchor)
        positive_embed = model(positive)
        negative_embed = model(negative)

        loss = criterion(anchor_embed, positive_embed, negative_embed)

        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for anchor, positive, negative in tqdm(test_loader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)

            loss = criterion(anchor_embed, positive_embed, negative_embed)
            total_loss += loss.item()

    return total_loss / len(test_loader)

def get_embedding(img_url):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingModel().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True, map_location=torch.device(device)))
    model.eval()

    with torch.no_grad():
        image = get_PIL_image(img_url)
        image = transform(image).unsqueeze(0).to(device)
        embedding = model(image).squeeze(0).cpu()
        embedding = embedding.numpy().tolist()

    return embedding    
