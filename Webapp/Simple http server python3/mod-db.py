"""
import sqlite3

# Initialize the SQLite database
conn = sqlite3.connect('data.db')
c = conn.cursor()

# Create the submissions table with a date column
c.execute('''CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY,
    topic TEXT,
    text TEXT,
    date TEXT
)''')

# Create the users table
c.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE,
    password TEXT
)''')

conn.commit()
"""

import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('data.db')
c = conn.cursor()

# Drop the existing submissions table if it exists
c.execute('''DROP TABLE IF EXISTS submissions''')

# Create the submissions table with the date column
c.execute('''
CREATE TABLE submissions (
    id INTEGER PRIMARY KEY,
    topic TEXT,
    text TEXT,
    date TEXT
)
''')

# Create the users table (if it doesn't already exist)
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE,
    password TEXT
)
''')

conn.commit()
conn.close()
