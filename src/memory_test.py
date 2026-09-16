from memory import init_db, search_memory


init_db()

results = search_memory("SQLite")

for result in results:
    print(result)