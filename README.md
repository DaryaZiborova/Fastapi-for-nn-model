# README 

## Требования

1. Python 3.8 и выше.
2. Установите необходимые зависимости:
   ```bash
   pip install -r requirements.txt
   ```
---

## Запуск (локально)

1. Установите библиотеку uvicorn:
    ```bash
     pip install uvicorn
     ```
2. Для запуска API выполните команду:
    ```bash
    uvicorn main:app --reload
    ```
После запуска API будет доступно по адресу: `127.0.0.1:8000`.

Тестовые запросы удобнее всего делать с эндпоинта `127.0.0.1:8000/docs`.

---

## Эндпоинты

### 1. **Тренировка модели**
`PUT /mm/api/v1/train-stylist-model/`

Запускает процесс обучения модели на основе триплетной выборки и сохраняет новую версию модели после обучения.

#### Параметры
- `first_article` (string, необязательный): Артикул товара из рекомендаций, с которого начинается обучение. Если не передать ничего, модель будет тренироваться на всех доступных рекомендациях.
- `triplets_per_ancor` (int, необязательный, по умолчанию: 100): Количество триплетов на одну рекомендацию.
- `num_epochs` (int, необязательный, по умолчанию: 10): Количество эпох обучения.

Количество элементов в выборке данных считается следующим образом:
n = triplets_per_ancor * количество рекомендаций, на которых будет обучаться модель

#### Пример
```bash
curl -X PUT "http://127.0.0.1:8000/mm/api/v1/train-stylist-model/" \
-H "Content-Type: application/json" \
-d '{
  "first_article": "61060642060058",
  "triplets_per_ancor": 50,
  "num_epochs": 5
}'
```

---

### 2. **Создание эмбеддингов фото товара**
`GET /mm/api/v1/create-image-embedding/`
Генерирует эмбеддинг для URL изображения.

#### Параметры
- `img_url` (string, обязательный): URL изображения (принимает как гиперссылки, так и ссылки вида `mm/images/images/имя.jpg`).

#### Ответ
`{
  "error": "Описание ошибки, если произошла", 
  "embedding": [Массив из 128-ми чисел с плавающей точкой / Пустой массив, если произошла ошибка]
}`

#### Пример
```bash
curl -X GET "http://127.0.0.1:8000/mm/api/v1/create-image-embedding/" \
-H "Content-Type: application/json" \
-d '{
  "img_url": "mm/images/images/sample.jpg"
}'
```

---

### 3. **Получение подходящих по стилю товаров**
`GET /mm/api/v1/get-suiting-products/`
Возвращает список артикулов товаров, подходящих под товар с переданным артикулом.

#### Параметры
- `article` (string, обязательный): Артикул товара.
- `top_n` (int, обязательный): Число, определяющее, сколько самых подходящих товаров необходимо найти.

#### Ответ
`{
  "error": "Описание ошибки, если произошла", 
  "articles": [Список артикулов в количестве top_n / Пустой список, если произошла ошибка]
}`

#### Пример
```bash
curl -X GET "http://127.0.0.1:8000/mm/api/v1/get-suiting-products/" \
-H "Content-Type: application/json" \
-d '{
  "article": "61060642060058",
  "top_n": 5
}'
```

---

## Структура проекта

- `api.py`: Основной файл API.
- `model.py`: Содержит модель, функции обучения и создания эмбеддингов.
- `utils.py`: Функции для работы с рекомендациями, категориями и изображениями.
- `db_conn.py`: Функции для работы с базой данных.

---

## Примечания

1. Убедитесь, что путь к обученной модели корректно указан в переменной `MODEL_SAVE_PATH`.
2. Для корректной работы эндпоинта для выбора наиболее подходящих товаров требуется предварительно настроенная и заполненная база данных.
На момент публикации в БД находится 1000 товаров.
### 3. При глобальном дообучении модели необходимо переписать все эмбеддинги изображений.

---

## Настройка БД, создание таблицы и вставка данных с PostgreSQL и pgvector 

1. Убедитесь, что облачная платформа, на которой развертывается база данных, поддерживает установку пользовательсих расширений.
2. Создание, настройка и подключение к базе данных.
3. Установка расширения pgvector.
```
CREATE EXTENSION IF NOT EXISTS vector;
SELECT * FROM pg_extension WHERE extname = 'vector';
```
Если результат запроса выглядит примерно как:
`(16470, 'vector', 10, 2200, True, '0.7.0', None, None)`
То расширение установлено успешно.

4. Создание таблицы.
```
CREATE TABLE IF NOT EXISTS products (
   product_id INTEGER PRIMARY KEY,
   article TEXT NOT NULL,
   main_photo TEXT,
   photo_embedding VECTOR(128) 
);
```
Вектор обязательно должен быть длиной 128.

5. Вставка данных.
```
INSERT INTO products (product_id, article, main_photo, photo_embedding)
VALUES (1234567, '1234567890', 'mm/images/images/1101184306003.jpg', {1.7102417,-0.14390856, ..., -6.090672})
ON CONFLICT (product_id) DO NOTHING;
```
Опять же, вектор должен быть длиной 128 чисел.

6. Пример на python.
- Установка psycopg2:
```bash
   pip install psycopg2-binary
```
- Скрипт:
```
import psycopg2

try:
    connection = psycopg2.connect(
        host = "ваш_host"
        port = 5432  # стандартный порт PostgreSQL
        database = "ваша_база_данных"
        username = "ваше_имя_пользователя"
        password = "ваш_пароль"
    )
    print("Connection to PostgreSQL database successful!")
    cursor = connection.cursor()

    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    connection.commit()
    print("pgvector extension installed successfully.")

    cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
    result = cursor.fetchone()
    if result:
        print(f"pgvector is enabled: {result}")
    else:
        print("Failed to enable pgvector.")

    cursor.execute("DROP TABLE IF EXISTS products;")
    connection.commit()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY,
        article TEXT NOT NULL,
        main_photo TEXT,
        photo_embedding VECTOR(128) 
    );
    """
    cursor.execute(create_table_query)
    connection.commit()
    print("Table 'products' created successfully.")

    insert_query = """
    INSERT INTO products (product_id, article, main_photo, photo_embedding)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (product_id) DO NOTHING;
    """
    for item in products_and_embeddings:  # Предпологая, что products_and_embeddings - это словарь с подходящими данными
        cursor.execute(
            insert_query,
            (
             item["product_id"],  # int
             item["article"],  # string
             item["main_photo"],  # string
             item["embedding"],  # массив из 128 чисел
            )
        )
    connection.commit()
    print("Data inserted successfully into 'products' table.")

except Exception as error:
    print(f"Error connecting to the database: {error}")
```
