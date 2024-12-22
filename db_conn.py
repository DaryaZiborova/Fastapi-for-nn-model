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

        if not embedding_result or embedding_result[0] == None:
            return articles

        search_query = "SELECT article FROM products ORDER BY photo_embedding <-> %s::vector LIMIT %s;"
        cursor.execute(search_query, (embedding_result[0], top_n))

        results = cursor.fetchall()
        articles = [row[0] for row in results]

    except Exception as error:
        return f"Error: {error}"

    finally:
        if connection:
            cursor.close()
            connection.close()
            return articles
        else:
            return "Connection to the database hasn't been succesful"
        
