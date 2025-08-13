import sqlite3


conn = sqlite3.connect("db/axiom.db")  # Creates or opens the database file
cursor = conn.cursor()

cursor.execute("SELECT * FROM sections s")
print(cursor.fetchall())
conn.close()