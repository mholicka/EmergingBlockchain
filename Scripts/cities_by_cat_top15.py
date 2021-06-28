import pandas as pd
from collections import ChainMap
import os

# ES imports
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, MultiSearch
from elasticsearch_dsl import A, Q

# pd.options.display.float_format = '{:,.2f}'.format

# set up ES
es = Elasticsearch(["http://localhost:9200"])


df_list = []


def ES_Search(searchRow):
    global es
    global df_list

    print(searchRow)
    firm_cat = searchRow["Firm Category"]
    number_firms = searchRow["Number of Firms"]
    print(firm_cat, number_firms)
    # return 0

    s = Search(using=es).index("categories*")

    # filter queries
    s = s.filter("match_phrase", cat_firm=firm_cat)

    # filter on date range
    s = s.filter(
        "range",
        **{
            "Date Announced": {
                "gte": "2009-01-01T05:00:00.688Z",
                "lte": "2019-01-01T00:00:00.829Z",
                "format": "strict_date_optional_time",
            }
        },
    )

    # Aggregations
    # s.aggs.bucket("super_region", "terms", field="super_region_firm", size=6).bucket("country", "terms", field="country-name_firm", size=1,order={'invested.value':'desc'}).metric(
    #     "invested", "cardinality", field="Invested Company"
    # )

    s.aggs.bucket(
        "super_region",
        "terms",
        field="super_region_firm",
        size=15,
        order={"invested": "desc"},
    )
    s.aggs["super_region"].metric("invested", "cardinality", field="Invested Company")
    s.aggs["super_region"].bucket("country", "terms", field="country-name_firm", size=1)

    print("####REQUEST#####")
    print(s.to_dict())
    print("####REQUEST#####")

    response = s.execute()

    # get agg, make into dataframe
    resp_agg = response.aggregations.super_region.buckets

    print(resp_agg)

    resp_dict = {}
    for hit in resp_agg:
        print(hit)
        info = hit.country.buckets[0]
        print(info)
        name = f"{hit.key}|{info.key}"
        resp_dict[name] = hit.invested.value
        # # print(hit.key,info.key,info.invested.value)

    print("####RESPONSE#####")
    print(resp_dict)
    print("####RESPONSE#####")

    col_name = "Firm Count"
    # print(resp_dict,type(resp_dict))
    resp_df = (
        pd.DataFrame.from_dict(resp_dict, orient="index")
        .reset_index()
        .rename(columns={"index": "Place", 0: col_name})
    )

    resp_df[["City Region", "Country"]] = resp_df.Place.str.split("|", expand=True)
    resp_df.sort_values(col_name, ascending=False, inplace=True)
    resp_df = resp_df[["City Region", "Country", col_name]]

    # print(resp_df)

    ## add in the total number of firms
    resp_df["TotFirms"] = number_firms
    resp_df["%"] = resp_df[col_name] / resp_df["TotFirms"] * 100

    final_dict = {"category": firm_cat, "df": resp_df}
    df_list.append(final_dict)

    print(resp_df)
    print("\n\n")
    return True


# for hit in response:
#     print(hit)

data_folder = "../Data/top15_Resources"

# define locations
category_csv = f"{data_folder}/CategoryMetrics.csv"
city_csv = f"{data_folder}/Main-Firms_perCityRegion.csv"

# load
cat_df = pd.read_csv(category_csv)
city_df = pd.read_csv(city_csv)

cat_df.apply(ES_Search, axis=1)


writer = pd.ExcelWriter(os.path.join(data_folder, "top15cities_by_category.xlsx"))

n_row = 1
n_col = 0
n_posList = 1
for item in df_list:
    item_df = item["df"]
    item_df.to_excel(
        writer, "cities_by_category", startcol=n_col, startrow=n_row + 1, index=False
    )
    sheet = writer.sheets["cities_by_category"]
    sheet.write(n_row, n_col, f"{n_posList}-{item['category']}")
    n_row += len(item_df.index) + 3
    n_posList += 1
    if n_row > 45:
        n_row = 1
        n_col += 6

writer.save()
