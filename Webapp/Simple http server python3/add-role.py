import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('data.db')
c = conn.cursor()

# Add the 'role' column to the 'users' table
c.execute("ALTER TABLE users ADD COLUMN role TEXT")

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database updated: 'role' column added.")
