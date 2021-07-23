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

from viral.base import (
    mk_and_save_covid_data,
    update_covid_data,
    instructions_to_make_bar_chart_race,
)

if __name__ == "__main__":
    import argh

    argh.dispatch_commands(
        [mk_and_save_covid_data, update_covid_data, instructions_to_make_bar_chart_race]
    )
