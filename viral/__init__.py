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
from py2store import kv_wrap, ZipReader  # google it and pip install it
from py2store.caching import mk_cached_store
from py2store import QuickPickleStore
from py2store.sources import FuncReader

DFLT_DATA_SOURCES_DIR = '~/ddir/my_sources'


def get_data_sources(data_sources=DFLT_DATA_SOURCES_DIR):
    if isinstance(data_sources, str):
        my_local_cache = fullpath(data_sources)
        CachedFuncReader = mk_cached_store(FuncReader,
                                           cache=QuickPickleStore(my_local_cache))
        data_sources = CachedFuncReader([country_flag_image_url,
                                         kaggle_coronavirus_dataset,
                                         city_population_in_time])
    assert isinstance(data_sources, Mapping)
    return data_sources


def country_flag_image_url():
    import pandas as pd
    return pd.read_csv(
        'https://raw.githubusercontent.com/i2mint/examples/master/data'
        '/country_flag_image_url.csv')


def kaggle_coronavirus_dataset():
    import kaggle
    from io import BytesIO
    # didn't find the pure binary download function, so using temp dir to emulate
    from tempfile import mkdtemp
    download_dir = mkdtemp()
    filename = 'novel-corona-virus-2019-dataset.zip'
    zip_file = os.path.join(download_dir, filename)

    dataset = 'sudalairajkumar/novel-corona-virus-2019-dataset'
    kaggle.api.dataset_download_files(dataset, download_dir)
    with open(zip_file, 'rb') as fp:
        b = fp.read()
    return BytesIO(b)


def city_population_in_time():
    import pandas as pd
    return pd.read_csv(
        'https://gist.githubusercontent.com/johnburnmurdoch/'
        '4199dbe55095c3e13de8d5b2e5e5307a/raw/fa018b25c24b7b5f47fd0568937ff6c04e384786'
        '/city_populations'
    )


def country_flag_image_url_prep(df: pd.DataFrame):
    # delete the region col (we don't need it)
    del df['region']
    # rewriting a few (not all) of the country names to match those found in kaggle
    # covid data
    # Note: The list is not complete! Add to it as needed
    old_and_new = [('USA', 'US'),
                   ('Iran, Islamic Rep.', 'Iran'),
                   ('UK', 'United Kingdom'),
                   ('Korea, Rep.', 'Korea, South')]
    for old, new in old_and_new:
        df['country'] = df['country'].replace(old, new)

    return df


@kv_wrap.outcoming_vals(
    lambda x: pd.read_csv(BytesIO(x)))  # this is to format the data as a dataframe
class ZippedCsvs(ZipReader):
    pass


# equivalent to ZippedCsvs = kv_wrap.outcoming_vals(lambda x: pd.read_csv(BytesIO(
# x)))(ZipReader)


# To update the coronavirus data:
def update_covid_data(data_sources=DFLT_DATA_SOURCES_DIR):
    """update the coronavirus data"""
    data_sources = get_data_sources(data_sources)
    if 'kaggle_coronavirus_dataset' in data_sources._caching_store:
        del data_sources._caching_store[
            'kaggle_coronavirus_dataset']  # delete the cached item
    _ = data_sources['kaggle_coronavirus_dataset']


def print_if_verbose(verbose, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def country_data_for_data_kind(data_sources=DFLT_DATA_SOURCES_DIR, kind='confirmed',
                               skip_first_days=0, verbose=False):
    """kind can be 'confirmed', 'deaths', 'confirmed_US', 'confirmed_US', 'recovered'"""
    data_sources = get_data_sources(data_sources)
    covid_datasets = ZippedCsvs(data_sources['kaggle_coronavirus_dataset'])

    df = covid_datasets[f'time_series_covid_19_{kind}.csv']
    # df = s['time_series_covid_19_deaths.csv']
    if 'Province/State' in df.columns:
        df.loc[df[
                   'Province/State'].isna(), 'Province/State'] = 'n/a'  # to avoid
        # problems arising from NaNs

    print_if_verbose(verbose, f"Before data shape: {df.shape}")

    # drop some columns we don't need
    p = re.compile('\d+/\d+/\d+')

    assert all(isinstance(x, str) for x in df.columns)
    date_cols = [x for x in df.columns if p.match(x)]
    if not kind.endswith('US'):
        df = df.loc[:, ['Country/Region'] + date_cols]
        # group countries and sum up the contributions of their states/regions/pargs
        df['country'] = df.pop('Country/Region')
        df = df.groupby('country').sum()
    else:
        df = df.loc[:, ['Province_State'] + date_cols]
        df['state'] = df.pop('Province_State')
        df = df.groupby('state').sum()

    print_if_verbose(verbose, f"After data shape: {df.shape}")
    df = df.iloc[:, skip_first_days:]

    if not kind.endswith('US'):
        # Joining with the country image urls and saving as an xls
        country_image_url = country_flag_image_url_prep(
            data_sources['country_flag_image_url'])
        t = df.copy()
        t.columns = [str(x)[:10] for x in t.columns]
        t = t.reset_index(drop=False)
        t = country_image_url.merge(t, how='outer')
        t = t.set_index('country')
        df = t
    else:
        pass

    return df


def mk_and_save_country_data_for_data_kind(data_sources=DFLT_DATA_SOURCES_DIR,
                                           kind='confirmed',
                                           save_dirpath='.',
                                           skip_first_days=0, verbose=False):
    data_sources = get_data_sources(data_sources)
    t = country_data_for_data_kind(data_sources, kind, skip_first_days, verbose)
    filepath = f'country_covid_{kind}.xlsx'
    if save_dirpath is not None:
        filepath = os.path.join(save_dirpath, filepath)
    t.to_excel(filepath)
    print_if_verbose(verbose, f"Was saved here: {filepath}")


def fullpath(path):
    return os.path.abspath(os.path.expanduser(path))


def mk_and_save_covid_data(data_sources=DFLT_DATA_SOURCES_DIR,
                           kinds=('confirmed', 'deaths', 'confirmed_US', 'deaths_US'),
                           save_dirpath='.', skip_first_days=39, verbose=True):
    """

    :param data_sources: Dirpath or py2store Store where the data is
    :param kinds: The kinds of data you want to compute and save
    :param skip_first_days:
    :param verbose:
    :return:
    """
    data_sources = get_data_sources(data_sources)
    for kind in kinds:
        mk_and_save_country_data_for_data_kind(data_sources, kind=kind,
                                               save_dirpath=save_dirpath,
                                               skip_first_days=skip_first_days,
                                               verbose=verbose)


def instructions_to_make_bar_chart_race():
    print("""
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

    - Confirmed cases (by country): https://public.flourish.studio/visualisation/1704821/
    - Deaths (by country): https://public.flourish.studio/visualisation/1705644/
    - US Confirmed cases (by state): https://public.flourish.studio/visualisation
    /1794768/
    - US Deaths (by state): https://public.flourish.studio/visualisation/1794797/
    """)


if __name__ == '__main__':
    import argh

    argh.dispatch_commands(
        [mk_and_save_covid_data, update_covid_data, instructions_to_make_bar_chart_race])
