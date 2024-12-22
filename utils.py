from PIL import Image
import requests
from io import BytesIO

def get_PIL_image(input_url):
    url = input_url if "https://chereggio.online/" in input_url else f"https://chereggio.online/{input_url}"

    try:
        response = requests.get(url)
        response.raise_for_status()

        image_data = BytesIO(response.content)
        image = Image.open(image_data).convert("RGB")
        return image

    except requests.RequestException as e:
        print(f"Error fetching the image from {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing the image: {e}")
        return None
    
def get_category(article):
    url = f"https://chereggio.online/mm/api/v1/product?article={article}"
    headers = {
        "Authorization": "Bearer a3f7fcb2b31d07fde1cef8cf26c269aa"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data.get("result", {}).get("error") == 0:
                return data.get("result", {}).get("product", {}).get("class")
            else:
                print(f"API error: {data.get('result', {}).get('error')}")
        else:
            print(f"HTTP error: {response.status_code}")
    except requests.RequestException as e:
        print(f"Request error: {e}")

    return None

def get_recommends(last_article=None):
    url = "https://chereggio.online/mm/api/v1/products_with_recommends"
    headers = {
        "Authorization": "Bearer a3f7fcb2b31d07fde1cef8cf26c269aa"
    }
    if last_article == None:
        params = {
            "limit": 10000,
        }
    else:
        params = {
            "limit": 10000,
            "last_article": last_article
        }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        recommends_data = response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return []

    
    recommends = recommends_data.get('result', {}).get('products', [])

    recommends_list = []
    for product in recommends:
        transformed = {
            "product_id": product["product_id"],
            "article": product["product_article"],
            "photo": product["product_photos"][0],
            "category": get_category(product["product_article"]),
            "recommends": [
                recommend["product_photos"][0]
                for recommend in product.get("recommends", [])
                if recommend.get("product_photos")
            ]
        }
        recommends_list.append(transformed)

    return recommends_list

def get_photos_with_categories():
    url = "https://chereggio.online/mm/api/v1/products"
    headers = {
        "Authorization": "Bearer a3f7fcb2b31d07fde1cef8cf26c269aa"
    }
    params = {
        "limit": 100000,
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")

    products = data.get('result', {}).get('products', [])
    photos_with_categories = {item["main_photo"]: item["class"] for item in products if item["main_photo"]}

    return photos_with_categories