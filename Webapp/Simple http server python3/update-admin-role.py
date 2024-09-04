import sqlite3
import bcrypt

# Connect to the SQLite database
conn = sqlite3.connect('data.db')
c = conn.cursor()

# Insert a new user with a role
username = 'admin2'
password = 'admin'
role = 'a'  # Use 'admin' for admin roles

# Hash the password
hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Insert the user into the database
c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
          (username, hashed_password.decode('utf-8'), role))

# Commit changes and close the connection
conn.commit()
conn.close()

print(f"User '{username}' added with role '{role}'.")
