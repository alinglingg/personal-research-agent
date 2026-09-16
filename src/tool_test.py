from tools import search_notes


results = search_notes("memory")

for result in results:
    print(result)
