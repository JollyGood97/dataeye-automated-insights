import ast
import pandas as pd
import sqlite3


def fetch_categorical_values(db, table_name, categorical_columns):
    categorical_values = {}
    for column in categorical_columns:
        values_query = f"SELECT DISTINCT `{column}` FROM `{table_name}`"
        values_str = db.run(values_query)
        try:
            # Ensure the result is properly formatted and evaluated
            values = ast.literal_eval(values_str)
            # Extract the distinct values and store them
            categorical_values[column] = [value[0] for value in values]
        except Exception as e:
            print(f"Error processing values for {column}: {e}")
            categorical_values[column] = []

    return categorical_values


def identify_categorical_columns(db, table_name, threshold=10):
    categorical_columns = []
    structure_str = db.run(f"PRAGMA table_info('{table_name}')")
    try:
        structure = ast.literal_eval(structure_str)
    except (ValueError, SyntaxError) as e:
        print(f"Error evaluating table structure: {e}")
        return []

    for column in structure:
        column_name = column[1]  # Extract the column name
        try:
            unique_count_str = db.run(
                f"SELECT COUNT(DISTINCT `{column_name}`) FROM `{table_name}`"
            )
            # Ensure the result is properly formatted and evaluated
            unique_count_result = ast.literal_eval(unique_count_str)
            # Extract the first element (the count) and ensure it's an integer
            unique_count = int(unique_count_result[0][0])
        except (ValueError, SyntaxError, TypeError) as e:
            print(f"Error processing unique count for {column_name}: {e}")
            continue

        if unique_count <= threshold:
            categorical_columns.append(column_name)

    return categorical_columns


def csv_to_sqlite(df, table_name, db_name="datastore.db"):
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_name)

    # Convert the DataFrame into a SQL table within the SQLite database
    df.to_sql(table_name, conn, if_exists="replace", index=False)

    # Close the database connection
    conn.close()

    return f"Table '{table_name}' has been created in database '{db_name}'."


def delete_table(table_name, db_name="datastore.db"):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Attempt to delete the specified table
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        return f"Table '{table_name}' has been deleted from database '{db_name}'."
    except sqlite3.Error as e:
        return f"Error deleting table '{table_name}': {e}"
    finally:
        # Close the database connection
        conn.close()


def view_all_tables(db_name="datastore.db", return_tables=False):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Retrieve the list of all tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Optionally return the list of tables instead of printing
    if return_tables:
        return [table[0] for table in tables]

    # Print each table name for command-line interface
    for table in tables:
        print(table[0])

    # Close the database connection
    conn.close()


def view_table_schema(table_name, db_name="datastore.db", return_schema=False):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # SQL query to get the schema of a table
    cursor.execute(f"PRAGMA table_info({table_name});")
    schema = cursor.fetchall()

    # Optionally return the schema instead of printing
    if return_schema:
        return [(column[1], column[2]) for column in schema]

    # Print the schema for command-line interface
    for column in schema:
        print(f"Column: {column[1]}, Type: {column[2]}")

    # Close the connection
    conn.close()
