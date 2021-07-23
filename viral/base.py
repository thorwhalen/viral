"""This module will help you make the data behind these:

- Confirmed cases (by country): https://public.flourish.studio/visualisation/1704821/
- Deaths (by country): https://public.flourish.studio/visualisation/1705644/
- US Confirmed cases (by state): https://public.flourish.studio/visualisation/1794768/
- US Deaths (by state): https://public.flourish.studio/visualisation/1794797/
"""

# # Getting and preparing the data

# Corona virus data here: https://www.kaggle.com/sudalairajkumar/novel-corona-virus
# -2019-dataset
# (direct download: https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019
# -dataset/download).
# It's currently updated daily, so download a fresh copy if you want.
#
# Population data here: http://api.worldbank.org/v2/en/indicator/SP.POP.TOTL
# ?downloadformat=csv
#
# It comes under the form of a zip file (currently named
# `novel-corona-virus-2019-dataset.zip`
# with several `.csv` files in them. We use `py2store` (To install: `pip install
# py2store`.
# Project lives here: https://github.com/i2mint/py2store) to access and pre-prepare it.
# It allows us to not have to unzip the file and replace the older folder with it
# every time we download a new one.
# It also gives us the csvs as `pandas.DataFrame` already.

import os
import re
import pandas as pd
from collections.abc import Mapping
from io import BytesIO

from haggle.dacc import KaggleBytesDatasetReader
from py2store import kv_wrap, ZipReader  # google it and pip install it
from py2store.caching import mk_cached_store
from py2store import QuickPickleStore
from py2store.sources import FuncReader

DATA_SOURCES_DIR_ENV_NAME = "viral_sources_dir"
DFLT_DATA_SOURCES_DIR = os.environ.get("viral_sources_dir", "~/ddir/my_sources")
DFLT_DATA_SOURCES_DIR = os.path.expanduser(DFLT_DATA_SOURCES_DIR)

kaggle_data = KaggleBytesDatasetReader()

if not os.path.isdir(DFLT_DATA_SOURCES_DIR):
    raise NotADirectoryError(
        f"""
    Directory doesn't exist: {DFLT_DATA_SOURCES_DIR} 
    Make the directory or change the viral_sources_dir environment variable to 
    point to a different one.
    """
    )


def country_flag_image_url():
    return pd.read_csv(
        "https://raw.githubusercontent.com/i2mint/examples/master/data"
        "/country_flag_image_url.csv"
    )


def _generate_state_flag_urls(states=None):
    """The code I used to make a csv of state flag urls.
    I then did
        print('\n'.join(_generate_state_flag_urls()))
    and copied the csv to:
    https://raw.githubusercontent.com/i2mint/examples/master/data/us_state_flag_url.csv
    """
    if states is None:
        data_sources = get_data_sources()
        uspopu = data_sources["us_states_population"]
        states = list(uspopu.index)

    yield "state,flag_image_url"
    from graze import Graze

    g = Graze()

    url_prefix = "https://github.com/CivilServiceUSA/us-states/raw/master/images/flags/"
    state_to_url = lambda x: f"{url_prefix}{x.lower().replace(' ', '-')}-small.png"
    for state in uspopu.index:
        state_image_url = state_to_url(state)
        try:
            g[state_image_url]
            yield f"{state},{state_image_url}"
        except KeyError:  # if you can't get it, it's not a valid url (only exception
            # is district of colombia (D.C.))
            pass

    # Manually add Washington DC since CivilServiceUSA doesn't include it
    yield f"District of Columbia," f"https://cdn11.bigcommerce.com/s-e2nupsxogj/product_images/uploaded_images" f"/washdc-nylon.png"


def state_flag_image_url():
    return pd.read_csv(
        "https://raw.githubusercontent.com/i2mint/examples/master/data"
        "/us_state_flag_url.csv"
    )


def kaggle_coronavirus_dataset():
    return kaggle_data["sudalairajkumar/novel-corona-virus-2019-dataset"]


def _raw_world_population_dataset():
    s = ZippedCsvs(kaggle_data["eng0mohamed0nabil/population-by-country-2020"])
    return s["population_by_country_2020.csv"]


def world_population_dataset():
    popu = _raw_world_population_dataset()

    # change the names of some countries to match covid naming
    covid_country_of_popu_country = {
        "South Korea": "Korea, South",
        "Saint Kitts & Nevis": "Saint Kitts and Nevis",
        "Taiwan": "Taiwan*",
        "Czech Republic (Czechia)": "Czechia",
        "Sao Tome & Principe": "Sao Tome and Principe",
        "United States": "US",
        "Morocco": "Kosovo",
        "Côte d'Ivoire": "Cote d'Ivoire",
    }

    t = list()
    for country in popu["Country (or dependency)"]:
        t.append(covid_country_of_popu_country.get(country, country))

    popu["country"] = t

    return popu


def _analyze_differences_of_countries_in_popu_and_covid_data():
    covid_datasets = ZippedCsvs(data_sources["kaggle_coronavirus_dataset"])
    t = covid_datasets["time_series_covid_19_confirmed.csv"]
    print(f"covid countries: {len(t)}")

    popu = _raw_world_population_dataset()
    covid_countries = t["Country/Region"]
    popu_countries = popu["Country (or dependency)"]
    print(
        f"in covid but not in popu:\n\t"
        f"{set(covid_countries).difference(popu_countries)}"
    )
    print(
        f"in popu but not in covid:\n\t{set(popu_countries).difference(covid_countries)}"
    )


def _raw_us_states_population():
    s = ZippedCsvs(kaggle_data["peretzcohen/2019-census-us-population-data-by-state"])
    return s["2019_Census_US_Population_Data_By_State_Lat_Long.csv"]


def us_states_population():
    return (
        _raw_us_states_population()
        .rename(columns={"STATE": "state", "POPESTIMATE2019": "population"})
        .set_index("state")["population"]
    )


def city_population_in_time():
    import pandas as pd

    return pd.read_csv(
        "https://gist.githubusercontent.com/johnburnmurdoch/"
        "4199dbe55095c3e13de8d5b2e5e5307a/raw/fa018b25c24b7b5f47fd0568937ff6c04e384786"
        "/city_populations"
    )


def country_flag_image_url_prep(df: pd.DataFrame):
    # delete the region col (we don't need it)
    del df["region"]
    # rewriting a few (not all) of the country names to match those found in kaggle
    # covid data
    # Note: The list is not complete! Add to it as needed
    old_and_new = [
        ("USA", "US"),
        ("Iran, Islamic Rep.", "Iran"),
        ("UK", "United Kingdom"),
        ("Korea, Rep.", "Korea, South"),
    ]
    for old, new in old_and_new:
        df["country"] = df["country"].replace(old, new)

    return df


@kv_wrap.outcoming_vals(
    lambda x: pd.read_csv(BytesIO(x))
)  # this is to format the data as a dataframe
class ZippedCsvs(ZipReader):
    pass


# equivalent to ZippedCsvs = kv_wrap.outcoming_vals(lambda x: pd.read_csv(BytesIO(
# x)))(ZipReader)


# To update the coronavirus data:
def update_covid_data(data_sources=DFLT_DATA_SOURCES_DIR):
    """update the coronavirus data"""
    data_sources = get_data_sources(data_sources)
    if "kaggle_coronavirus_dataset" in data_sources._caching_store:
        del data_sources._caching_store[
            "kaggle_coronavirus_dataset"
        ]  # delete the cached item
    _ = data_sources["kaggle_coronavirus_dataset"]


def print_if_verbose(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def _country_data_for_data_kind(
    data_sources=DFLT_DATA_SOURCES_DIR,
    kind="confirmed",
    per_capita=False,
    skip_first_days=0,
    verbose=False,
):
    """kind can be
    'confirmed', 'deaths', 'recovered', 'confirmed_US', 'confirmed_US'
    'confirmed_per_capita', 'deaths_per_capita', 'recovered_per_capita'
    """
    data_sources = get_data_sources(data_sources)
    covid_datasets = ZippedCsvs(data_sources["kaggle_coronavirus_dataset"])

    df = covid_datasets[f"time_series_covid_19_{kind}.csv"]
    # df = s['time_series_covid_19_deaths.csv']
    if "Province/State" in df.columns:
        # to avoid problems arising from NaNs
        df.loc[df["Province/State"].isna(), "Province/State"] = "n/a"  # to avoid

    population_by_country = None

    # if per_capita:
    #     popu = data_sources["world_population_dataset"]
    #     df = pd.merge(
    #         df,
    #         popu[["country", "Population (2020)"]],
    #         left_on="Country/Region",
    #         right_on="country",
    #     )
    #     # df = df.set_index('country')
    #     population_by_country = df.set_index("country")["Population (2020)"]
    #     del df["Population (2020)"]

    print_if_verbose(verbose, f"Before data shape: {df.shape}")

    # drop some columns we don't need
    p = re.compile("\d+/\d+/\d+")

    assert all(isinstance(x, str) for x in df.columns)
    date_cols = [x for x in df.columns if p.match(x)]

    if not kind.endswith("US"):
        df = df.loc[:, ["Country/Region"] + date_cols]
        # group countries and sum up the contributions of their states/regions/pargs
        df["country"] = df.pop("Country/Region")
        df = df.groupby("country").sum()

        if per_capita:
            popu = data_sources["world_population_dataset"]
            df = pd.merge(df, popu[["country", "Population (2020)"]], on="country")
            df = df.set_index("country")
            population_by_country = df["Population (2020)"]
            del df["Population (2020)"]
    else:
        df = df.loc[:, ["Province_State"] + date_cols]
        df["state"] = df.pop("Province_State")
        df = df.groupby("state").sum()

    print_if_verbose(verbose, f"After data shape: {df.shape}")
    df = df.iloc[:, skip_first_days:]

    # normalize by population if requested
    if per_capita:
        df = 1e6 * df.div(population_by_country, axis="rows")

    # Add flag url if not US
    if not kind.endswith("US"):
        # Joining with the country image urls and saving as an xls
        country_image_url = country_flag_image_url_prep(
            data_sources["country_flag_image_url"]
        )
        t = df.copy()
        t.columns = [str(x)[:10] for x in t.columns]
        t = t.reset_index(drop=False)
        t = country_image_url.merge(t, how="outer")
        # t = t.set_index("country")
        df = t
    else:
        pass

    return df


def get_covid_base_data(data_sources=DFLT_DATA_SOURCES_DIR, kind="confirmed"):
    data_sources = get_data_sources(data_sources)
    covid_datasets = ZippedCsvs(data_sources["kaggle_coronavirus_dataset"])

    df = covid_datasets[f"time_series_covid_19_{kind}.csv"]
    # df = s['time_series_covid_19_deaths.csv']
    if "Province/State" in df.columns:
        # to avoid problems arising from NaNs
        df.loc[df["Province/State"].isna(), "Province/State"] = "n/a"  # to avoid

    return df


def add_flag_urls(df, image_urls):
    df.columns = [str(x)[:10] for x in df.columns]
    df = df.reset_index(drop=False)
    return image_urls.merge(df, how="outer")


def data_for_data_kind(
    data_sources=DFLT_DATA_SOURCES_DIR,
    kind="confirmed",
    per_capita=False,
    skip_first_days=0,
    verbose=False,
):
    if kind.endswith("US"):
        return us_data_for_data_kind(
            data_sources, kind, per_capita, skip_first_days, verbose
        )
    else:
        return country_data_for_data_kind(
            data_sources, kind, per_capita, skip_first_days, verbose
        )


def country_data_for_data_kind(
    data_sources=DFLT_DATA_SOURCES_DIR,
    kind="confirmed",
    per_capita=False,
    skip_first_days=0,
    verbose=False,
):
    """kind can be
    'confirmed_US', 'deaths_US', 'recovered_US'
    """
    data_sources = get_data_sources(data_sources)
    assert not kind.endswith("US")

    df = get_covid_base_data(data_sources, kind)

    print_if_verbose(verbose, f"Before data shape: {df.shape}")

    if per_capita:
        popu = data_sources["world_population_dataset"]

        df = pd.merge(
            df,
            popu[["country", "Population (2020)"]],
            left_on="Country/Region",
            right_on="country",
        )
        # df = df.set_index('country')
        population = df.set_index("country")["Population (2020)"]
        del df["Population (2020)"]

    # drop some columns we don't need
    p = re.compile("\d+/\d+/\d+")

    assert all(isinstance(x, str) for x in df.columns)
    date_cols = [x for x in df.columns if p.match(x)]

    df = df.loc[:, ["Country/Region"] + date_cols]
    # group countries and sum up the contributions of their states/regions/pargs
    df["country"] = df.pop("Country/Region")
    df = df.groupby("country").sum()

    if per_capita:
        popu = data_sources["world_population_dataset"]
        df = pd.merge(df, popu[["country", "Population (2020)"]], on="country")
        df = df.set_index("country")
        population = df["Population (2020)"]
        del df["Population (2020)"]

    print_if_verbose(verbose, f"After data shape: {df.shape}")
    df = df.iloc[:, skip_first_days:]

    # normalize by population if requested
    if per_capita:
        df = 1e6 * df.div(population, axis="rows")
        df = df.dropna(axis=0, how="any")

    return add_flag_urls(
        df, country_flag_image_url_prep(data_sources["country_flag_image_url"])
    )


def us_data_for_data_kind(
    data_sources=DFLT_DATA_SOURCES_DIR,
    kind="confirmed",
    per_capita=False,
    skip_first_days=0,
    verbose=False,
):
    """kind can be
    'confirmed_US', 'deaths_US', 'recovered_US'
    """
    assert kind.endswith("US")

    data_sources = get_data_sources(data_sources)
    df = get_covid_base_data(data_sources, kind)

    print_if_verbose(verbose, f"Before data shape: {df.shape}")

    # drop some columns we don't need
    p = re.compile("\d+/\d+/\d+")

    assert all(isinstance(x, str) for x in df.columns)
    date_cols = [x for x in df.columns if p.match(x)]

    df = df.loc[:, ["Province_State"] + date_cols]
    df["state"] = df.pop("Province_State")
    df = df.groupby("state").sum()

    # normalize by population if requested
    if per_capita:
        population = data_sources["us_states_population"]
        df = 1e6 * df.div(population, axis="rows")
        df = df.dropna(axis=0, how="any")

    print_if_verbose(verbose, f"After data shape: {df.shape}")
    df = df.iloc[:, skip_first_days:]

    return add_flag_urls(df, data_sources["state_flag_image_url"])


def mk_and_save_country_data_for_data_kind(
    data_sources=DFLT_DATA_SOURCES_DIR,
    kind="confirmed",
    per_capita=False,
    save_dirpath=".",
    skip_first_days=0,
    verbose=False,
):
    data_sources = get_data_sources(data_sources)
    t = data_for_data_kind(data_sources, kind, per_capita, skip_first_days, verbose)
    per_capita_str = ""
    if per_capita:
        per_capita_str = "_per_capita"
    filepath = f"country_covid_{kind}{per_capita_str}.xlsx"
    if save_dirpath is not None:
        filepath = os.path.join(save_dirpath, filepath)
    t.to_excel(filepath)
    print_if_verbose(verbose, f"Was saved here: {filepath}")


def fullpath(path):
    return os.path.abspath(os.path.expanduser(path))


def mk_and_save_covid_data(
    data_sources=DFLT_DATA_SOURCES_DIR,
    kinds=("confirmed", "deaths", "confirmed_US", "deaths_US"),
    per_capita=(False, True),
    save_dirpath=".",
    skip_first_days=39,
    verbose=True,
):
    """

    :param data_sources: Dirpath or py2store Store where the data is
    :param kinds: The kinds of data you want to compute and save
    :param skip_first_days:
    :param verbose:
    :return:
    """
    data_sources = get_data_sources(data_sources)
    for kind in kinds:
        for _per_capita in per_capita:
            mk_and_save_country_data_for_data_kind(
                data_sources,
                kind=kind,
                per_capita=_per_capita,
                save_dirpath=save_dirpath,
                skip_first_days=skip_first_days,
                verbose=verbose,
            )


def instructions_to_make_bar_chart_race():
    print(
        """
    Prep: Go to https://public.flourish.studio/ and get a free account.
    (1) Make the data (use mk_and_save_covid_data for example)
    (2) (a) If you're creating a new graph: 
            Go to https://app.flourish.studio/templates
            Choose "Bar chart race". At the time of writing this, it was here: 
            https://app.flourish.studio/visualisation/1706060/
        (b) If you're updating a graph, go to it
    (3) Upload the data you've created (or updated) in (1)
    (4) Play with the settings to get it right

    Here are examples of graphs I've generated with this module:

    - Country cases: https://public.flourish.studio/visualisation/1704821/
    - Country Deaths: https://public.flourish.studio/visualisation/1705644/
    - US State cases: https://public.flourish.studio/visualisation/1794768/
    - US State Deaths: https://public.flourish.studio/visualisation/1794797/
    
    - Country cases per capita: https://public.flourish.studio/visualisation/6812015/
    - Country deaths per capita: https://public.flourish.studio/visualisation/6812118/
    - US State Cases per capita: https://public.flourish.studio/visualisation/6811967/
    - US State deaths per capita: https://public.flourish.studio/visualisation/6812165
    
    """
    )


# ---------------------------------------------------------------------------------------
# A root store to provide the different data sources.

# Uses FuncReader, whose keys are function names and values are what they return
# when called (without arguments)
# Uses CachedFuncReader to cache the results.


def get_data_sources(data_sources=DFLT_DATA_SOURCES_DIR):
    if isinstance(data_sources, str):
        my_local_cache = fullpath(data_sources)
        CachedFuncReader = mk_cached_store(
            FuncReader, cache=QuickPickleStore(my_local_cache)
        )
        data_sources = CachedFuncReader(
            [
                kaggle_coronavirus_dataset,
                world_population_dataset,
                us_states_population,
                country_flag_image_url,
                state_flag_image_url,
                city_population_in_time,
            ]
        )
    assert isinstance(data_sources, Mapping)
    return data_sources


data_sources = get_data_sources()

# if __name__ == "__main__":
#     country_data_for_data_kind(kind="confirmed", per_capita=True)
