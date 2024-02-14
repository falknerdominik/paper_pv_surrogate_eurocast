import requests
from prefect import flow, task

from pv_surrogate_eurocast.constants import GeoData, Paths


@task
def download_from_url(url: str, target: str):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Write the content of the response (which is the zip file) to the specified path
        with open(GeoData.natural_earth_data, "wb") as file:
            file.write(response.content)
    else:
        raise Exception(f"Request failed with status code {response.status_code}")


@task
def setup_structure():
    Paths.ensure_directories_exists()


@flow
def main():
    setup_structure()
    download_from_url(
        url=GeoData.natural_earth_data_url,
        target=GeoData.natural_earth_data,
    )


if __name__ == "__main__":
    main()
