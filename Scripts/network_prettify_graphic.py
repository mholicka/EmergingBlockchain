# %%
import pandas as pd
import pathlib
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap as Basemap
from sklearn.preprocessing import MinMaxScaler, data
import numpy as np
from curved_edges import curved_edges
from matplotlib.collections import LineCollection
import cmasher as cmr
import sys, os
from pprint import pprint
import matplotlib.colors as mcol
from jenkspy import JenksNaturalBreaks
import matplotlib.colors as mcolors
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import ticker


# for Notebook
# %matplotlib inline


current_path = pathlib.Path(__file__).parent.absolute().parent

# Config variables

encode = "utf-8"
offset = 15
plt_w = 7
plt_h = 5
weight_threshold = 10
scale_weight = {"min": 0.3, "max": 1.5}
full_world = True
# colourmap sub map,
cmap_name = "hsv"
cmap_range = (0.65, 1)
cmap_def = cmr.get_sub_cmap(cmap_name, cmap_range[0], cmap_range[1])

node_col = "maroon"
node_sz = 1.6

# label_dict = ["Very Low", "Low", "Medium", "High", "Very High", "Extreme"]
label_dict = ["Low", "Low2", "Medium", "High", "Very High2", "Very High"]


# remove country codes from the labels
no_countries = True

# the connections between US Cities and SV are too high, even when scaled,
# removing it makes the map way nicer.
# They will have thier own map

remove_US_Cities = True
remove_non_US_cities = False

keep_cities = ["Silicon Valley", "New York"]

if remove_US_Cities and remove_non_US_cities:
    print(
        "You have selected to remove both US and non-US cities.. Ambiguous... try again"
    )
    sys.exit()
elif remove_US_Cities:
    remove_non_US_cities = False
    file_name = "US_sansSV_removed"
    offset = 15
    proj = "merc"
elif remove_non_US_cities:
    remove_US_Cities = False
    file_name = "US_ONLY"
    offset = 5
    proj = "aea"


# file locations
world_cities_csv = f"{current_path}/Data/worldcities.csv"
edge_list_csv = f"{current_path}/Data/City_edgelist_prettify.csv"
node_list_csv = f"{current_path}/Data/cities_nodes.csv"
node_geo_csv = f"{current_path}/Data/nodelist_withgeo.csv"

save_thesis = True


if save_thesis:
    save_path_wm = f"{current_path}/Output/Centrality/WorldMap_thesis.png"
    fmt_save = "png"
    save_dpi = 600
    plt_w = 5
    plt_h = 3
    fig = plt.figure(figsize=(plt_w, plt_h))

else:
    save_path_wm = f"{current_path}/Output/Centrality/WorldMap_800dpi.asfd"
    fmt_save = "png"
    save_dpi = 800
    plt.figure(figsize=(11.69, 8.27))

ax = plt.axes((0.1, 0.1, 0.5, 0.8))
m_res = "i"


# reading in DF
edge_df = pd.read_csv(edge_list_csv, encoding=encode)
node_df = pd.read_csv(node_list_csv, encoding=encode)


# manual intervention was needed to assign some items
pre_manual = False

# https://stackoverflow.com/questions/58048415/how-to-increase-label-spacing-to-avoid-label-overlap-in-networkx-graph
def label_no_overlap(pos_graph):

    test_case = ("Hong Kong", "Shanghai")
    test_case_good = ("Singapore", "Hong Kong")
    pos_1 = pos_graph

    # 17 - London, 14- Zug, 71- Paris

    for p in pos_1:
        # London label should be moved up
        if p == 17.0:
            print(p, pos_1[p])
            pos_1[p] = (pos_1[p][0] + 250000, pos_1[p][1] + 250000)
            print(f"changed : {pos_1[p]}")
        elif p == 71.0:
            pos_1[p] = (pos_1[p][0] + 500000, pos_1[p][1] + 100000)
        elif p == 14.00:
            pos_1[p] = (pos_1[p][0] + 600000, pos_1[p][1])
        elif p == 26.0:
            pos_1[p] = (pos_1[p][0] + 600000, pos_1[p][1] - 250000)

    # print (pos_1)
    # print("return")
    return pos_1


def make_netx_graph(graph, type_g, main_g, keep_cites_fun=keep_cities):

    curr_nodes = graph.nodes
    SG_node_attrs = [(n, main_g.nodes[n]) for n in curr_nodes]
    graph.add_nodes_from(SG_node_attrs)

    if type_g == "All":
        US_Cites = [
            n
            for n, d in graph.nodes(data=True)
            if (not d["super_region"] in keep_cites_fun and d["country_code"] == "USA")
        ]
        graph.remove_nodes_from(US_Cites)
    elif type_g == "US":
        non_US_Cites = [
            n for n, d in graph.nodes(data=True) if not d["country_code"] == "USA"
        ]
        graph.remove_nodes_from(non_US_Cites)
    else:
        pass

    return graph


def createMappingDict(graph, mapping_dict_in):
    # list of nodes
    l_nodes = list(graph.nodes)

    # re-create the mapping dict to suit the new nodes.
    mapping_dict_new = dict(
        (k, mapping_dict_in[k]) for k in l_nodes if k in mapping_dict_in
    )
    return mapping_dict_new


def makeGEO(graph, geo_df):
    # get all nodes not in the graph for plotting
    graph_nodes = [int(x) for x in list(graph.nodes)]
    not_in = geo_df.loc[geo_df.index.drop(graph_nodes)]
    # print (not_in)

    # Grab the lats/ lons for the nodes in the graph
    lat_nodes = nx.get_node_attributes(graph, "latitude")
    lon_nodes = nx.get_node_attributes(graph, "longitude")

    # create the bounding box dictionary based on the lat/lon of the values
    # and adding an offset to make it look nice
    bb = {
        "lat": {
            "min": min(lat_nodes.values()) - offset,
            "max": max(lat_nodes.values()) + offset,
        },
        "lon": {
            "min": min(lon_nodes.values()) - offset,
            "max": max(lon_nodes.values()) + offset,
        },
    }

    world_bb = {
        "lat": {"min": -48.92001097, "max": 67.52181866000001},
        "lon": {"min": -138.1216442, "max": 154.7514074},
    }

    if full_world:
        bb = world_bb

    print("Creating basemap...")

    m = Basemap(
        projection=proj,
        llcrnrlat=bb["lat"]["min"],
        urcrnrlat=bb["lat"]["max"],
        llcrnrlon=bb["lon"]["min"],
        urcrnrlon=bb["lon"]["max"],
        resolution=m_res,
    )

    # m.shadedrelief(scale=0.5)
    m.fillcontinents(color="lightgray", lake_color="lightblue", zorder=0, alpha=0.1)
    m.drawcountries(linewidth=0.1)
    m.drawstates(linewidth=0.1)
    m.drawcoastlines(linewidth=0.1)

    scatter_x, scatter_y = m(not_in["longitude"], not_in["latitude"])

    plt.plot(scatter_x, scatter_y, "bo", color="maroon", markersize=0.8)

    print("Basemap created")

    # place the objects on the map based on the lon and lat of the nodes
    mx, my = m(list(lon_nodes.values()), list(lat_nodes.values()))
    pos = {}
    for count, elem in enumerate(list(lon_nodes.keys())):
        pos[elem] = (mx[count], my[count])

    # print(bb)
    return pos


def getWeightScale(sub_graph, scale_weight_dict, lab=label_dict):

    node_data = sub_graph.nodes(data=True)

    # weights_list = nx.get_edge_attributes(sub_graph, 'weight')

    df = nx.to_pandas_edgelist(sub_graph)
    df["source"] = df["source"].map(lambda x: node_data[x]["super_region"])
    df["target"] = df["target"].map(lambda x: node_data[x]["super_region"])

    # Get the weights of the edges, reshape to a 1D array.
    weights = list(nx.get_edge_attributes(sub_graph, "weight").values())

    # natural breaks, jenks
    jnb = JenksNaturalBreaks()
    jnb.fit(weights)
    # try:
    #     print(jnb.labels_)
    #     print(jnb.groups_)
    #     # print(jnb.inner_breaks_)
    # except:
    #     pass
    df["Class"] = jnb.labels_
    binned_w = jnb.labels_

    weights_arr = np.array([binned_w]).reshape(-1, 1)
    # scale weights between the thresholds using MinMax scaler
    scaler = MinMaxScaler(
        feature_range=(scale_weight_dict["min"], scale_weight_dict["max"])
    )
    new_weights = scaler.fit_transform(weights_arr)
    new_weights = list(new_weights.ravel())
    print(new_weights)

    # manual assignments
    # binned_w = pd.cut(df["weight"],bins = bins,right =False,include_lowest=True,labels=labels)

    df["Class"] = binned_w
    df["Weight"] = new_weights
    df["Label"] = df["Class"].map(lambda x: lab[x])
    # df.to_csv()
    print(df)
    df.to_csv(
        f"{current_path}/Data/weighted_labels/weighted_with_labels_WT_{weight_threshold}.csv",
        index=False,
    )

    return new_weights


if pre_manual:
    wcities_df = pd.read_csv(world_cities_csv, encoding=encode)
    # Geocoding cities
    wcities_df = wcities_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    geo_df = pd.merge(
        node_df,
        wcities_df,
        how="left",
        left_on=["super_region", "country_code"],
        right_on=["nameascii", "adm0_a3"],
    )
    geo_df.to_csv(f"{current_path}/nodelist_withgeo.csv", index=False, encoding=encode)
else:

    # Read in the node DF, drop any empty rows in the node and edge DFs
    node_geo_df = pd.read_csv(node_geo_csv)
    node_geo_df.dropna(subset=["latitude", "longitude"], how="any", inplace=True)
    edge_df.dropna(how="any", inplace=True)

    # Create the ID field from the index
    node_geo_df = node_geo_df.reset_index().rename(columns={"index": "ID"})

    node_geo_df.to_csv(
        f"{current_path}/Data/inc_nodelist_withgeo.csv", index=True, encoding=encode
    )

    # keep only certain columns
    keep_cols = ["country_code", "super_region", "ID"]

    # ensure the strings will work together
    edge_df = edge_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Create the edge df with merges to node df, both to and from.
    edge_df_to = pd.merge(
        edge_df,
        node_geo_df[keep_cols],
        how="left",
        left_on=["country_code_firm", "super_region_firm"],
        right_on=["country_code", "super_region"],
    )
    edge_df_to.rename(columns={"ID": "node_to"}, inplace=True)

    edge_df_from = pd.merge(
        edge_df_to,
        node_geo_df[keep_cols],
        how="left",
        left_on=["country_code_invest", "super_region_invest"],
        right_on=["country_code", "super_region"],
    )
    edge_df_from.rename(columns={"ID": "node_from"}, inplace=True)

    # create the final edge df, and set the weight to 1 for each
    edge_df_final = edge_df_from
    edge_df_final.to_csv(f"{current_path}/Data/Edgelist_edited.csv", index=True)

    edge_df_final.dropna(how="any", inplace=True)
    edge_df_final["weight"] = 1

    # calculate edge weights and store into a new df
    weighted = (
        edge_df_final.groupby([edge_df_final["node_from"], edge_df_final["node_to"]])
        .sum()
        .reset_index()
    )

    ##### Begin of Networkx Items##############

    # creating the label
    node_geo_df["label"] = (
        node_geo_df["nameascii"].astype(str)
        + "-"
        + node_geo_df["country_code"].astype(str)
    )

    # ID to label dict
    mapping_dict = dict(zip(node_geo_df["ID"], node_geo_df["label"]))
    mapping_dict_no_countries = dict(zip(node_geo_df["ID"], node_geo_df["nameascii"]))

    mapping_dict_select = mapping_dict_no_countries if no_countries else mapping_dict

    node_attr = node_geo_df.set_index("ID").to_dict("index")
    geo_id_df = node_geo_df.set_index("ID")
    # print (weighted[ (weighted['node_from']==16) | (weighted['node_to']==16)])

    # Create the network with the edgelist, set the attributes, and remove all self loops
    G = nx.from_pandas_edgelist(
        weighted,
        source="node_from",
        target="node_to",
        edge_attr=["weight"],
        create_using=nx.MultiGraph,
    )
    nx.set_node_attributes(G, node_attr)
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    # getting the largest connected component
    largest_cc = G.subgraph(max(nx.connected_components(G), key=len))

    # grabbing the subset graph based on the weight threshold
    subset = [
        (u, v, d)
        for u, v, d in largest_cc.edges(data=True)
        if d["weight"] >= weight_threshold
    ]
    SG = nx.Graph(subset)
    # SG = nx.Graph(largest_cc)

    # All Countries
    all_countries = make_netx_graph(graph=SG, type_g="", main_g=G)
    non_US_dict = createMappingDict(all_countries, mapping_dict_select)
    non_US_Pos = makeGEO(all_countries, geo_id_df)
    non_US_weight = getWeightScale(all_countries, scale_weight)
    print(f"classes : {non_US_weight}")

    base_cm = cmap_def

    cm1 = base_cm
    sm = plt.cm.ScalarMappable(
        cmap=cm1, norm=plt.Normalize(vmin=min(non_US_weight), vmax=max(non_US_weight))
    )
    test = sm.to_rgba(list(non_US_weight))

    node_collection = nx.draw_networkx_nodes(
        all_countries,
        non_US_Pos,
        node_size=node_sz,
        node_color=node_col,
        alpha=1,
        ax=ax,
    )
    node_collection.set_zorder(55)
    curves = curved_edges(all_countries, non_US_Pos)

    norm_input = plt.Normalize(vmin=min(non_US_weight), vmax=max(non_US_weight))
    # print(non_US_weight)

    lc_all = LineCollection(
        curves,
        cmap=base_cm,
        alpha=0.7,
        linewidths=non_US_weight,
        norm=norm_input,
        colors=test,
    )
    label_pos_all = label_no_overlap(non_US_Pos)
    # labels=non_US_dict,
    # nx.draw_networkx_labels(
    #     all_countries, pos=label_pos_all, font_size=4.5, alpha=1, ax=ax
    # )
    plt.gca().add_collection(lc_all)
    plt.gca().set_position([0, 0, 1, 1])

    # plt.tight_layout()
    # plt.show()
    # save the figure bins=bins_def,labels=labels_def

    tick_labs = np.unique(non_US_weight).tolist()
    cbaxes = inset_axes(
        ax,
        width="15%",
        height="1%",
        loc="lower center",
        bbox_to_anchor=(0.05, 0.03, 1, 1),
        bbox_transform=ax.transAxes,
    )
    # cbar = plt.colorbar(
    #     sm, orientation="horizontal", ticks=tick_labs, pad=0, shrink=0.2, cax=cbaxes
    # )
    cbar = plt.colorbar(sm, orientation="horizontal", pad=0, shrink=0.2, cax=cbaxes)
    # cbar.set_ticks([])
    cbar.ax.set_xticklabels(label_dict)  # horizontal colorbar
    cbar.set_ticks([cbar.vmin, cbar.vmax])
    cbar.set_ticklabels([label_dict[0], label_dict[-1]])
    cbar.outline.set_linewidth(0.5)

    # # cbar.ax.tick_params(axis="x", which="major", pad=-1)
    cbar.ax.tick_params(labelsize=3, pad=-0.2)

    # ax.set_title(
    #     f"Node Color : {node_col}, node size : {node_sz} \n Size : {plt_w}w x {plt_h}h",
    #     fontsize=5,
    # )

    save_path_wm = f"{current_path}/Output/final_iters/wm_{node_col}_{node_sz}_{plt_w}w_x_{plt_h}h_manual.{fmt_save}"

    # print(f"Saving figure to {save_path_wm} ")
    # plt.savefig(save_path_wm, dpi=save_dpi, format=fmt_save, rasterize=True)
    # print("Done Saving")
    plt.show()

    # sys.exit()

    #### EIGENVECTOR CALCULATION ###############################
    ### compute eigenvector centrality and plot it.

    # we want to put back the countries
    non_US_dict = createMappingDict(all_countries, mapping_dict)

    def relabel(pos, x_shift, y_shift):

        return {n: (x + x_shift, y + y_shift) for n, (x, y) in pos.items()}
        # pos_3

    def slight_adjust(pos, code, x_factor, y_factor):
        pos[code] = pos[code] + np.array([x_factor, y_factor])
        return pos

    def draw(G, pos, measures, measure_name, extra_txt):

        # manual shifts
        # boston : 45.0
        pos = slight_adjust(pos, 45.0, 0.21, -0.03)
        # Beijing: 9.0
        pos = slight_adjust(pos, 9.0, -0.01, 0)

        pos_nodes = relabel(pos, 0, 0.1)

        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_size=250,
            cmap=base_cm,
            node_color=list(measures.values()),
            nodelist=measures.keys(),
            linewidths=1,
        )
        nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))
        nx.draw_networkx_labels(G, pos_nodes, labels=non_US_dict)
        nx.draw_networkx_edges(G, pos)
        # plt.title(
        #     f"EigenVector Centrality - Cities with Weight >= {weight_threshold} -  {extra_txt}"
        # )
        cb = plt.colorbar(nodes, orientation="horizontal", pad=0.02, shrink=0.5)
        cb.ax.set_xticklabels([0.2, 0.3, 0.4, 0.5, 0.6])

    centrality = nx.eigenvector_centrality(all_countries)
    # print(centrality)
    nx.set_node_attributes(all_countries, centrality, "eigen_cent")

    potentials = [929, 5823, 7306, 5140, 4045, 5908]
    # for idx in np.random.randint(low=1,high=9999, size=10):
    for idx in potentials:
        if not idx == 7306:
            continue
        # print (idx)
        plt.clf()
        plt.figure(figsize=(12, 10), edgecolor="black")
        pos = nx.spring_layout(
            all_countries,
            iterations=50,
            scale=1.5,
            seed=np.random.RandomState(idx),
            k=1 / math.sqrt(len(all_countries.nodes())),
        )
        draw(
            all_countries,
            pos,
            nx.eigenvector_centrality(all_countries),
            "eigen_cent",
            extra_txt=f"Seed:{idx}",
        )
        axis = plt.gca()
        # maybe smaller factors work as well, but 1.1 works fine for this minimal example
        axis.set_xlim([1.1 * x for x in axis.get_xlim()])
        axis.set_ylim([1.1 * y for y in axis.get_ylim()])
        plt.tight_layout()
        plt.minorticks_off()

        plt.savefig(
            f"{current_path}/Output/Centrality/EigenCentrality_thesis.jpg",
            dpi=800,
            format="jpg",
        )

        plt.show()


# %%
