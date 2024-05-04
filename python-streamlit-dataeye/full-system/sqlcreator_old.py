import pandas as pd
import sqlite3

csv_path = "../csvs/Marks.csv"


def csv_to_sqlite(csv_path, table_name, db_name="datastore.db"):
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_name)

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Convert the DataFrame into a SQL table within the SQLite database
    df.to_sql(table_name, conn, if_exists="replace", index=False)

    # Close the database connection
    conn.close()

    print(f"Table '{table_name}' has been created in database '{db_name}'.")


def delete_table(table_name, db_name="datastore.db"):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Attempt to delete the specified table
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        print(f"Table '{table_name}' has been deleted from database '{db_name}'.")
    except sqlite3.Error as e:
        print(f"Error deleting table '{table_name}':", e)
    finally:
        # Close the database connection
        conn.close()


def view_all_tables(db_name="datastore.db"):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Retrieve the list of all tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Print each table name
    print("Tables in the database:")
    for table in tables:
        print(table[0])

    # Close the database connection
    conn.close()


def view_table_schema(table_name, db_name="datastore.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # SQL query to get the schema of a table
    cursor.execute(f"PRAGMA table_info({table_name});")
    schema = cursor.fetchall()

    # Print the schema
    print(f"Schema of '{table_name}':")
    for column in schema:
        print(f"Column: {column[1]}, Type: {column[2]}")

    # Close the connection
    conn.close()


if __name__ == "__main__":
    csv_to_sqlite("../csvs/netflix_titles.csv", "NETFLIX_TITLES")
    # delete_table("NETFLIX_TITLES.csv")
    # view_all_tables()
    view_table_schema("NETFLIX_TITLES")
