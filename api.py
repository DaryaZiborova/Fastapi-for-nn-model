from fastapi import FastAPI
from torch.utils.data import DataLoader, random_split
import torch
from model import TripletDataset, TripletLoss, EmbeddingModel, test, train, get_embedding, transform, MODEL_SAVE_PATH
from utils import get_recommends, get_photos_with_categories
from db_conn import get_suiting_articles

app = FastAPI()

category_mismatch = {
    1: [2, 3, 8, 14, 16, 27, 44, 45, 48, 49, 55, 83, 97],  # Пальто
    2: [1, 3, 8, 14, 16, 27, 44, 45, 48, 49, 55, 83, 97],  # Плащ
    3: [1, 2, 8, 14, 16, 27, 44, 45, 48, 49, 55, 83, 97],  # Пальто+1
    4: [34, 36, 48, 49, 54, 56, 57, 80, 83, 86, 91, 92],  # Жакет
    7: [34, 36, 48, 49, 54, 56, 57, 80, 83, 86, 91, 92],  # Жакет+1
    8: [1, 2, 3, 14, 16, 27, 44, 45, 48, 49, 55, 83, 97],  # Полупальто
    10: [13, 14, 18, 22, 23, 24, 30, 32, 48, 49, 57, 62, 64, 77, 78, 83],  # Юбка
    11: [16, 19, 26, 31, 45, 48, 49, 54, 56, 57, 63, 83, 94, 95],  # Рубашка
    13: [10, 14, 18, 22, 23, 24, 30, 32, 48, 49, 57, 62, 64, 77, 78, 83],  # Брюки
    14: [10, 13, 18, 22, 23, 24, 30, 32, 48, 49, 57, 62, 64, 77, 78, 83],  # Шорты
    15: [34, 36, 48, 49, 54, 56, 57, 80, 83, 86, 91, 92],  # Жилет
    16: [1, 2, 4, 7, 8, 19, 26, 31, 45, 48, 49, 54, 56, 57, 62, 64, 94, 95, 97],  # Топ
    18: [10, 13, 14, 22, 23, 24, 30, 32, 48, 49, 57, 62, 64, 77, 78, 83],  # Джинсы
    19: [11, 16, 26, 31, 45, 48, 49, 54, 56, 57, 63, 83, 94, 95],  # Блузон
    22: [10, 13, 14, 18, 24, 30, 32, 48, 49, 57, 62, 63, 64, 77, 78, 83],  # Платье
    23: [10, 13, 14, 18, 24, 30, 32, 48, 49, 57, 62, 63, 64, 77, 78, 83],  # Платье+1
    24: [10, 13, 14, 18, 22, 23, 30, 32, 48, 49, 57, 62, 63, 77, 78, 83],  # Комбинезон
    26: [11, 16, 19, 31, 45, 48, 49, 54, 56, 57, 63, 83, 94, 95],  # Двойка
    27: [1, 2, 3, 8, 14, 16, 44, 45, 48, 49, 55, 83, 97],  # Пальто-жилет
    30: [13, 14, 18, 22, 23, 24, 10, 32, 48, 49, 57, 62, 64, 77, 78, 83],  # Юбка трикотажная
    31: [11, 16, 19, 26, 45, 48, 49, 54, 56, 57, 63, 83, 94, 95],  # Блузон трикотажный
    32: [10, 13, 14, 18, 24, 30, 48, 49, 57, 62, 63, 64, 77, 78, 83],  # Платье трикотажное
    33: [10, 13, 14, 18, 22, 23, 24, 30, 32, 48, 49, 57, 62, 64, 77, 78, 83],  # Брюки трикотажные
    34: [36, 48, 49, 54, 55, 56, 57, 83, 86, 91, 92],  # Кардиган
    35: [34, 36, 48, 49, 54, 56, 57, 80, 83, 86, 91, 92],  # Жилет трикотажный
    36: [38, 48, 49, 55, 83, 86, 91, 92],  # Свитер
    37: [48, 49, 54, 55, 83, 86],  # Накидка трикотажная
    38: [36, 48, 49, 55, 83, 86, 91, 92],  # Двойка трикотажная
    40: [13, 14, 18, 22, 23, 24, 30, 32, 48, 49, 57, 62, 64, 77, 78, 83],  # Юбка кожаная
    41: [1, 2, 3, 8, 14, 16, 44, 45, 48, 49, 55, 83, 97],  # Дубленка
    42: [10, 13, 14, 18, 24, 30, 32, 48, 49, 57, 62, 63, 64, 77, 78, 83],  # Платье кожаное
    43: [10, 13, 14, 18, 22, 23, 24, 30, 32, 48, 49, 57, 62, 64, 77, 78, 83],  # Брюки кожаные
    44: [1, 2, 3, 8, 14, 16, 44, 45, 48, 49, 55, 83, 97],  # Жакет кожаный
    45: [1, 2, 3, 8, 14, 16, 44, 47, 48, 49, 55, 83, 97],  # Полупальто искусственный мех
    46: [1, 2, 3, 8, 14, 16, 44, 47, 48, 49, 55, 83, 95, 97],  # Мех
    47: [1, 2, 3, 8, 14, 16, 44, 45, 48, 49, 55, 83, 97],  # Плащ кожаный
    48: [1, 2, 3, 8, 14, 16, 44, 45, 49, 55, 83, 95, 97],  # Куртка стеганая
    49: [1, 2, 3, 8, 14, 16, 44, 45, 48, 55, 83, 95, 97],  # Пальто стеганое
    50: [45, 48, 49, 54, 55, 56, 83, 86],  # Ремень
    51: [86, 55],  # Сумки
    52: [76, 79, 52, 86],  # Обувь
    54: [86, 55, 83, 95, 97, 14, 16, 61],  # Шарф
    55: [73, 80, 83, 99],  # Носки
    56: [86, 55, 83, 95, 97, 14, 16],  # Перчатки
    57: [4, 11, 13, 14, 16, 15, 35, 19, 86, 94],  # Шапка
    58: [86, 55, 83, 95, 97, 14, 16, 61],  # Воротник меховой
    60: [73, 80, 83, 99],  # Промо
    61: [86, 55, 83, 95, 97, 14, 16, 54],  # Воротник 
    62: [10, 13, 14, 18, 24, 30, 32, 48, 49, 57, 63, 64, 77, 78, 83],  # Платье-джерси
    63: [10, 13, 14, 18, 24, 30, 32, 48, 49, 57, 62, 64, 77, 78, 83],  # Двойка-джерси
    64: [10, 13, 14, 18, 22, 23, 30, 32, 48, 49, 57, 62, 63, 77, 78, 83],  # Комбинезон
    71: [86, 55],  # Сумка
    73: [73, 99],  # Аксессуары
    75: [55],  # Бижутерия
    76: [45, 48, 49, 86, 79, 52, 54, 56],  # Сникерсы
    77: [13, 14, 18, 22, 23, 24, 30, 32, 48, 49, 57, 62, 64, 78, 83],  # Юбка-джерси
    78: [10, 13, 14, 18, 22, 23, 24, 30, 32, 33, 48, 49, 57, 62, 64, 77, 83],  # Брюки-джерси
    79: [14, 16, 76, 52, 83, 86],  # Сапоги
    80: [48, 49, 54, 55, 57, 58, 79],  # Очки
    81: [13, 18, 24, 33, 43, 45, 48, 49, 64, 78],  # Колготы
    83: [54, 45, 48, 49, 56, 57, 58, 86],  # Купальник
    86: [51, 52, 54, 45, 48, 49, 56, 57, 58, 50, 76, 79, 83],  # Белье
    90: [1, 2, 3, 8, 14, 16, 27, 44, 45, 48, 49, 55, 83, 97],  # Пальто-жакет
    91: [34, 36, 48, 49, 54, 56, 57, 80, 83, 86, 92],  # Жакет-джерси
    92: [4, 7, 16, 34, 36, 45, 48, 49, 56, 83, 86],  # Свитшот
    94: [11, 16, 19, 22, 23, 24, 26, 31, 32, 45, 48, 49, 54, 56, 57, 62, 63, 64, 83, 95, 97],  # Блуза
    95: [1, 2, 8, 11, 16, 19, 26, 31, 45, 48, 49, 54, 56, 57, 62, 64, 94, 97],  # Поло
    97: [4, 7, 1, 2, 8, 16, 19, 45, 48, 49, 54, 56, 57, 95],  # Футболка
    99: [73, 99],  # Аксессуар
}

@app.put("/mm/api/v1/train-stylist-model/")
def train_model(last_article: str = None, triplets_per_ancor: int = 100, num_epochs: int = 10):
    recommends = get_recommends(last_article)
    photos_with_categories = get_photos_with_categories()
    if len(recommends.get("recommends", [])) == 0:
        return {"error": recommends.get("error")}
    if len(photos_with_categories.get("photos_with_categories", [])) == 0:
        return {"error": photos_with_categories.get("error")}
    photos_with_categories = photos_with_categories.get("photos_with_categories", [])
    recommends = recommends.get("recommends", [])

    dataset = TripletDataset(category_mismatch, 
                             recommends, 
                             photos_with_categories,
                             transform=transform, 
                             triplets_per_anchor=triplets_per_ancor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingModel().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True, map_location=torch.device(device)))

    criterion = TripletLoss(margin=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    avg_train_loss = 0
    avg_test_loss = 0
    print('Starting to train the model...')
    for epoch in range(num_epochs):
        print(f"Training step {epoch+1}")
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        avg_train_loss += train_loss

        print(f"Testing step {epoch+1}")
        test_loss = test(model, test_dataloader, criterion, device)
        avg_test_loss += test_loss

        print(f"Epoch {epoch+1}/{num_epochs}, average train loss: {train_loss:.4f}, average test loss: {test_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    avg_train_loss /= num_epochs
    avg_test_loss /= num_epochs

    response = {"error": "No errors", "response":
                f"The model has been succesfully trained. Average train loss: {avg_train_loss:.3f}, average test loss: {avg_test_loss:.3f}"}
    return response

@app.get("/mm/api/v1/create-image-embedding/")
def create_embedding(img_url: str):
    embedding = get_embedding(img_url)
    return embedding


@app.get("/mm/api/v1/get-suiting-products/")
def get_suiting_products(article: str, top_n: int):
    suiting_articles = get_suiting_articles(article, top_n)
    return suiting_articles

