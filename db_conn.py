import psycopg2

host = "vectordb.c78quweqw7tg.us-east-1.rds.amazonaws.com"
port = 5432
database = "vectorDB"
username = "postgres"
password = "001001p4ssw0rd"

def get_suiting_articles(article, top_n):
    connection = None
    articles = []
    try:
        connection = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=username,
            password=password
        )
        cursor = connection.cursor()

        search_query = "SELECT photo_embedding FROM products WHERE article=%s"
        cursor.execute(search_query, (article,))
        embedding_result = cursor.fetchone()

        if not embedding_result:
            return {"error": "No such article in the database", "articles": []}
        if not embedding_result[0]:
            return {"error": "No image embedding found for provided article", "articles": []}

        search_query = "SELECT article FROM products ORDER BY photo_embedding <-> %s::vector LIMIT %s;"
        cursor.execute(search_query, (embedding_result[0], top_n))

        results = cursor.fetchall()
        articles = [row[0] for row in results]

        return {"error": "No errors", "articles": articles}
    
    except Exception as error:
        return {"error": error, "articles": []}

    finally:
        if connection:
            cursor.close()
            connection.close()
        else:
            return {"error": "Connection to the database hasn't been succesful", "articles": []}
        
