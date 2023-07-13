import pandas as pd


class DateFeatureEngineer:
    """
    This class handles feature engineering for date type variables.
    """

    def __init__(self, date_format: str="%m-%d-%Y"):
        """
        Constructor for the DateFeatureEngineer class.

        Parameters:
        date_format (str): The format of the date in the input data.
        """
        self.date_format = date_format

    def fit_transform(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Fit and transform the data.

        Parameters:
        df (pandas.DataFrame): The input data.
        column (str): The column in the dataframe to be transformed.
        
        Returns:
        df (pandas.DataFrame): The transformed data.
        """
        df = self._split_date(df, column)
        return df

    def _split_date(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Splits a date into separate features.

        Parameters:
        df (pandas.DataFrame): The input data.
        column (str): The column in the dataframe to be transformed.
        
        Returns:
        df (pandas.DataFrame): The transformed data.
        """
        df[column] = pd.to_datetime(df[column], format=self.date_format)
        df['month'] = df[column].dt.month
        df['year'] = df[column].dt.year
        return df



if __name__ == "__main__":
    # df = pd.read_csv("../data/retail_prices_encoded.csv")
    # # Create a DateFeatureEngineer instance
    # date_engineer = DateFeatureEngineer(date_format="%Y-%m-%d")
    # # Transform the DataFrame
    # df_transformed = date_engineer.fit_transform(df, "month_year")
    # df_transformed.to_csv("../data/retail_prices_encoded_date.csv", index=False)  
    pass 