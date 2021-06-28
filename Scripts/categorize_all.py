import pandas as pd
import os, sys
import statistics
import warnings
import numpy as np


keep_cols = [
    "Location",
    "category_groups_list",
    "country_code",
    "super_region",
    "country-name",
    "name",
    "short_description",
    "founded_on",
]


working_folder = "Data/Full_Cat_Assignment"

catDF = pd.read_csv(f"{working_folder}/categoryAssignment.csv")
catDF = catDF.apply(lambda col: col.str.lower())

firm_df = pd.read_csv(f"{working_folder}/companies_geoloced.csv")
firm_df = firm_df[keep_cols]

# print(firm_df)

# create category DF for quick access
catDict = dict(sorted(catDF.values.tolist()))

# print(catDict)

tiers = {
    "top": [
        "health",
        "natural resources and energy",
        "sports and entertainment",
        "personnel",
        "education",
        "property and real estate",
        "government and military",
        "logistics and transportation",
        "food and agriculture",
        "community and lifestyle",
    ],
    "mid": ["commerce and shopping", "privacy and security", "science and engineering"],
    "low": ["software and analytics", "finance", "hardware", "mobile"],
}


def clean(list_in_raw):
    # print(list_in_raw)
    list_in = list(map(lambda x: x.lower(), list_in_raw.split(",")))
    categories = [catDict[x].strip() for x in list_in]
    # print (categories)
    max_remove_FA = list(
        filter(lambda a: a not in ["software and analytics", "finance"], categories)
    )
    categories_clean = max_remove_FA if len(max_remove_FA) > 0 else categories
    categories_clean_unique = (
        set(max_remove_FA) if len(max_remove_FA) > 0 else set(categories)
    )

    try:
        suggestion = statistics.mode(categories_clean)
        # print (f"suggestion:{mode}")
    except:
        # check if the unique is only 1
        if len(categories_clean_unique) == 1:
            suggestion = categories_clean_unique[0]
        else:
            # $ suggestion_tier =[]
            for tier in tiers:
                print(tier)
                tier_list = [x for x in categories_clean_unique if x in tiers[tier]]
                print(tier_list)
                if len(tier_list) > 0:
                    suggestion = tier_list[0] if len(tier_list) == 1 else "Manual"
                    print(suggestion)
                    break

            # print(row['Invested Company'])
            # print (row['name_invest'])
            # print(f'{categories}=>{categories_clean}')
            # print(f'{categories}=>{categories_clean_unique} => {suggestion}')
            # input("Press Enter to continue...")

    if suggestion == "Manual":
        mobile_list = ["messaging and telecommunications", "mobile"]
        # manual classification
        if categories_clean_unique == {"finance", "software and analytics"}:
            suggestion = "software and analytics"
        elif (
            "commerce and shopping" in categories_clean_unique
            and len(categories_clean_unique) <= 2
        ):
            suggestion = "commerce and shopping"
        elif "administrative services" in list_in:
            suggestion = "personnel"
        elif "payments" in list_in:
            suggestion = "commerce and shopping"
        elif "real estate" in list_in:
            suggestion = "property and real estate"
        elif "government and military" in list_in:
            suggestion = "government and military"
        elif categories_clean_unique == {
            "science and engineering",
            "privacy and security",
        }:
            suggestion = "science and engineering"
        elif categories_clean_unique == {"personnel", "logistics and transportation"}:
            suggestion = "personnel"
        elif any(item in mobile_list for item in list_in):
            suggestion = "mobile"
        elif categories_clean_unique == {"health", "sports and entertainment"}:
            suggestion = "health"
        elif "agriculture and farming" in list_in:
            suggestion = "food and agriculture"
        elif "education" in categories_clean:
            suggestion = "education"

        print("#####MANUAL#########")
        print(list_in)
        print(f"{categories}=>{categories_clean} =>{categories_clean_unique}")

        f_e = ",".join(list_in)
        print(f"for Excel : {f_e}")
        print("#####MANUAL#########")
    print(f"{categories}=>{categories_clean_unique} => {suggestion}")
    print(suggestion)
    # print ('\n --------------------- \n')
    return suggestion


# firm_df = firm_df.head(50)

firm_df["category_groups_list"].replace("", np.nan, inplace=True)
firm_df.dropna(subset=["category_groups_list"], inplace=True)
firm_df["main_cat"] = firm_df["category_groups_list"].apply(clean)
firm_df_manual = firm_df[firm_df.main_cat == "Manual"]
print(firm_df_manual.shape)
# print(firm_df_manual)
firm_df.to_csv(f"{working_folder}/firm_categorizations_revised_2.csv")

# test ="Artificial Intelligence,Data and Analytics,Design,Health Care,Information Technology,Media and Entertainment,Science and Engineering,Software"
# test = "Energy,Information Technology,Manufacturing,Privacy and Security,Science and Engineering,Software,Transportation"
# test = "Financial Services,Information Technology,Payments,Real Estate,Software"
clean(test)
