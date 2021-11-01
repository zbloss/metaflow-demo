from metaflow import FlowSpec, step, conda

class IngestData(FlowSpec):
    @staticmethod
    def download_file(url: str, custom_filename: str = None):
        """
        Downloads file from url.

        Args:
            url (str): The HTTP URL pointer to the file you wish to download.
            custom_filename (str): Optional, overrides the downloaded filename.

        Returns:
            local_filename (str): The name of the downloaded file.
        """
        import os
        import shutil
        import requests
        from pathlib import Path

        if custom_filename is not None:
            local_filename = custom_filename
        else:
            local_filename = url.split("/")[-1]

        directory, _ = os.path.split(local_filename)
        Path(directory).mkdir(parents=True, exist_ok=True)

        with requests.get(url, stream=True) as r:
            with open(local_filename, "wb") as f:
                shutil.copyfileobj(r.raw, f)

        return local_filename

    @step
    def start(self):
        """
        Begins the data flow, defines the url and files to be downloaded.

        """

        self.base_url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg"
        )
        self.data_file = "auto-mpg.data"
        self.data_directory = "auto-mpg/data"
        self.feature_names = [
            "mpg"
            "cylinders"
            "displacement"
            "horsepower"
            "weight"
            "acceleration"
            "model year"
            "origin"
            "car name"
        ]
        self.next(self.get_file)

    @step
    def get_file(self):
        """
        Downloads all of the data file.

        """

        url = f"{self.base_url}/{self.data_file}"
        custom_filename = f"{self.data_directory}/raw/{self.data_file}"
        filename = self.download_file(url=url, custom_filename=custom_filename)
        print(f"Downlaoded {url} to {custom_filename}")
        self.next(self.process_dataset)

    @conda(libraries={"pandas": "1.3.4"})
    @step
    def process_dataset(self):
        """
        Loads the downloaded raw data and processes it into a
        pandas dataframe.

        """

        import os
        import pandas as pd

        with open(
            os.path.join(self.data_directory, "raw", self.data_file), "r"
        ) as in_file:
            rows = in_file.readlines()
            in_file.close()

        values = []
        for row in rows:
            features, car_name = row.replace("\n", "").split("\t")
            features = features.split()
            car_name = car_name.replace('"', "")
            car_name = car_name.replace("'", "")
            features.append(car_name)
            values.append(features)

        self.data = pd.DataFrame(values, columns=self.feature_names)
        for column in self.data.columns:
            if column != "car name":
                self.data[column] = self.data[column].astype(float)

        self.next(self.save_dataset)

    @step
    def save_dataset(self):
        """
        Saves the processed dataset under a templated path
        as a gzipped parquet file.

        """

        import os
        from datetime import datetime

        timestamp = datetime.now()
        yearkey = "yearkey={}".format(timestamp.strftime("%Y"))
        monthkey = "monthkey={}".format(timestamp.strftime("%m"))
        daykey = "daykey={}".format(timestamp.strftime("%d"))
        directory_to_save_processed_data_to = os.path.join(
            self.data_directory, "processed", yearkey, monthkey, daykey
        )
        if not os.path.exists(directory_to_save_processed_data_to):
            os.makedirs(directory_to_save_processed_data_to)

        self.data.to_parquet(
            os.path.join(directory_to_save_processed_data_to, "data.parquet.gzip"),
            compression="gzip",
        )
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    IngestData()
