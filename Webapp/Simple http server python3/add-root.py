import sqlite3
import bcrypt
import os
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

ADMIN_USERNAME = os.getenv('ADMIN_USERNAME')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD_HASH')

# Hash the admin password
hashed_password = bcrypt.hashpw(ADMIN_PASSWORD.encode('utf-8'), bcrypt.gensalt())

# Retry mechanism in case of database lock
retries = 5
while retries > 0:
    try:
        # Connect to SQLite database
        conn = sqlite3.connect('data.db')
        c = conn.cursor()

        # Insert the admin user into the users table
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (ADMIN_USERNAME, hashed_password.decode('utf-8')))
        conn.commit()
        print("Admin user added successfully!")
        conn.close()
        break
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e):
            print("Database is locked, retrying in 1 second...")
            time.sleep(1)
            retries -= 1
        else:
            print(f"An error occurred: {e}")
            break
    finally:
        if 'conn' in locals():
            conn.close()

if retries == 0:
    print("Failed to add admin user after multiple attempts due to database lock.")
