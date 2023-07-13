import pandas as pd
from sqlalchemy import create_engine


class DataLoader:
    """
    DataLoader class encapsulates the details of connecting to a database 
    and loading data into a pandas DataFrame.
    """

    def __init__(self, db_uri: str):
        """
        Initializes DataLoader class with a database URI and creates an SQLAlchemy engine.
        
        Parameters:
            db_uri (str): The database URI.
        """
        self.db_uri = db_uri
        self.engine = create_engine(self.db_uri)
        self.data = None

    def load_data(self, table_name: str) -> pd.DataFrame:
        """
        Loads data from the specified table into a DataFrame, which is stored 
        as an instance variable self.data.

        Parameters:
            table_name (str): The name of the table from which to load data.

        Returns:
            DataFrame: The loaded data.
        """
        query = "SELECT * FROM " + table_name
        self.data = pd.read_sql_query(query, self.engine)
        return self.data

    def get_data(self) -> pd.DataFrame:
        """
        Returns the loaded data. If no data has been loaded, it raises a ValueError.
        
        Returns:
            DataFrame: The loaded data.

        Raises:
            ValueError: If data has not been loaded yet.
        """
        if self.data is not None:
            return self.data
        else:
            raise ValueError("Data has not been loaded yet. Please call 'load_data' first.")

# data_loader = DataLoader('postgresql://postgres:ayushsingh@localhost:5432/cs001')
# data_loader.load_data('retail_prices')
# df = data_loader.get_data()
# print(df.head())