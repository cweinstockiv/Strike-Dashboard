import pandas as pd
import geopandas as gpd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import os
import glob

CRS_PROJECTED = "EPSG:3035"

# load ACLED data

df = pd.read_csv("https://drive.google.com/uc?export=download&id=1hSx0W2upPuTIB6nMnwv-cmBuS1q2eWg4")


# clean data

df.columns = df.columns.str.lower().str.strip()
df = df.loc[:, ~df.columns.duplicated()]

event_date_cols = [c for c in df.columns if "event_date" in c]
if len(event_date_cols) > 1:
    df = df.drop(columns=event_date_cols[1:])

df = df[df["geo_precision"] == 1]
df = df[df["time_precision"] == 1]

df["event_date"] = pd.to_datetime(df["event_date"],errors="coerce")

df["Day"] = df["event_date"].dt.day
df["Month"] = df["event_date"].dt.month
df["Year"] = df["event_date"].dt.year
df["Week in Year"] = df["event_date"].dt.strftime("%U").astype(int)

columns_to_drop = [
    "event_id_cnty","disorder_type","event_type","sub_event_type",
    "actor1","assoc_actor_1","inter1","interaction","civilian_targeting",
    "iso","region","admin1","admin2","admin3","location",
    "source","source_scale","fatalities","tags","timestamp"
]

df = df.drop(columns=columns_to_drop,errors="ignore")

df["country"] = df["country"].astype(str).str.strip()
df = df.dropna(subset=["latitude","longitude"])

print(f" Data cleaned.Rows remaining:{len(df)}")


# time series

df_ru = df[df["country"] == "Russia"]
df_ua = df[df["country"] == "Ukraine"]

weekly_ru = df_ru.resample("W",on="event_date").size()
weekly_ua = df_ua.resample("W",on="event_date").size()

weekly_df = pd.DataFrame({
    "Russia": weekly_ru,
    "Ukraine": weekly_ua
}).fillna(0)

weekly_df["Russia_4wk_avg"] = weekly_df["Russia"].rolling(4).mean()
weekly_df["Ukraine_4wk_avg"] = weekly_df["Ukraine"].rolling(4).mean()

monthly_ru = df_ru.resample("ME",on="event_date").size()
monthly_ua = df_ua.resample("ME",on="event_date").size()

monthly_df = pd.DataFrame({
    "Russia": monthly_ru,
    "Ukraine": monthly_ua
}).fillna(0)


# KDE

df["MonthLabel"] = df["event_date"].dt.to_period("M").astype(str)
months = sorted(df["MonthLabel"].unique())

kde_dropdown_options = [{"label":"Total Strikes","value":"ALL"}]
for m in months:
    kde_dropdown_options.append({"label":m,"value":m})


# Cube

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
).to_crs(CRS_PROJECTED)

gdf["x"] = gdf.geometry.x
gdf["y"] = gdf.geometry.y

gdf["year"] = gdf["event_date"].dt.year
gdf["week_num"] = gdf["event_date"].dt.isocalendar().week
gdf["week"] = gdf["year"].astype(str) + "-W" + gdf["week_num"].astype(str)
gdf["month"] = gdf["event_date"].dt.to_period("M").astype(str)

month_labels = sorted(gdf["month"].unique())
week_labels = sorted(gdf["week"].unique())


# dashboard

app = dash.Dash(__name__)

app.layout = html.Div([

    html.H1("Russia–Ukraine Strike Analysis Dashboard LAST UPDATED: 3/16/2026"),

    dcc.Tabs([


        # cube tab

        dcc.Tab(label="Space-Time Cube",children=[

            html.Label("Time Aggregation"),

            dcc.RadioItems(
                id="time_agg",
                options=[
                    {"label":"Monthly","value":"month"},
                    {"label":"Weekly","value":"week"}
                ],
                value="month",
                inline=True
            ),

            html.Br(),

            html.Label("Strike Selection"),

            dcc.Dropdown(
                id="country_mode",
                options=[
                    {"label":"Both","value":"both"},
                    {"label":"Ukraine Only","value":"Ukraine"},
                    {"label":"Russia Only","value":"Russia"}
                ],
                value="both",
                clearable=False
            ),

            html.Br(),

            html.Label("Bin Size (km)"),

            dcc.Slider(
                id="bin_size",
                min=10,
                max=100,
                step=10,
                value=20,
                marks={i:f"{i} km" for i in range(10,110,10)}
            ),

            html.Br(),

            html.Div(
                id="month_slider_container",
                children=[
                    dcc.RangeSlider(
                        id="month_slider",
                        min=0,
                        max=len(month_labels)-1,
                        step=1,
                        value=[0,len(month_labels)-1],
                        marks={i:month_labels[i] for i in range(len(month_labels))}
                    )
                ]
            ),

            html.Div(
                id="week_dropdown_container",
                children=[
                    dcc.Dropdown(
                        id="week_dropdown",
                        options=[{"label":w,"value":w} for w in week_labels],
                        value=week_labels[0],
                        clearable=False
                    )
                ],
                style={"display":"none"}
            ),

            dcc.Graph(id="cube_graph",style={"height":"900px"}),

            html.H3("Strike Map"),

            dcc.Graph(
                id="map_graph",
                style={"height":"600px"},
                config={"scrollZoom":True,"displaylogo":False}
            )

        ]),


        # time series tab

        dcc.Tab(label="Time Series",children=[

            dcc.Graph(
                figure=go.Figure([
                    go.Scatter(x=weekly_df.index,y=weekly_df["Russia"],name="Russia Weekly",line=dict(color="red")),
                    go.Scatter(x=weekly_df.index,y=weekly_df["Ukraine"],name="Ukraine Weekly",line=dict(color="blue")),
                    go.Scatter(x=weekly_df.index,y=weekly_df["Russia_4wk_avg"],name="Russia 4-Week Avg",line=dict(color="darkred",width=3)),
                    go.Scatter(x=weekly_df.index,y=weekly_df["Ukraine_4wk_avg"],name="Ukraine 4-Week Avg",line=dict(color="darkblue",width=3))
                ]).update_layout(title="Weekly Strike Activity",yaxis_title="Number of Strikes")
            ),

            dcc.Graph(
                figure=go.Figure([
                    go.Scatter(x=monthly_df.index,y=monthly_df["Russia"],mode="lines+markers",name="Russia"),
                    go.Scatter(x=monthly_df.index,y=monthly_df["Ukraine"],mode="lines+markers",name="Ukraine")
                ]).update_layout(title="Monthly Strike Activity",yaxis_title="Number of Strikes")
            )

        ]),


        # KDE tab

        dcc.Tab(label="KDE", children=[

            html.H2("Kernel Density Estimation",style={"textAlign":"center"}),

            html.Div([
                dcc.Dropdown(
                    id="month-filter",
                    options=kde_dropdown_options,
                    value="ALL",
                    clearable=False,
                    style={"width":"300px"}
                )
            ],style={"display":"flex","justifyContent":"center","marginBottom":"10px"}),

            dcc.Graph(
                id="kde-map",
                style={"height":"90vh"},
                config={"scrollZoom":True}
            )

        ])

    ])
])


# time control

@app.callback(
    Output("month_slider_container","style"),
    Output("week_dropdown_container","style"),
    Input("time_agg","value")
)
def toggle_time_controls(time_agg):

    if time_agg == "month":
        return {"display":"block"},{"display":"none"}

    return {"display":"none"},{"display":"block"}


# update cube and map

@app.callback(
    Output("cube_graph","figure"),
    Output("map_graph","figure"),
    Input("country_mode","value"),
    Input("time_agg","value"),
    Input("month_slider","value"),
    Input("week_dropdown","value"),
    Input("bin_size","value"),
    Input("cube_graph","clickData")
)
def update_cube(country_mode,time_agg,month_range,selected_week,bin_km,clickData):

    cell_size = bin_km * 1000
    gdf_local = gdf.copy()

    if time_agg == "month":
        labels = month_labels[month_range[0]:month_range[1]+1]
        filtered = gdf_local[gdf_local["month"].isin(labels)].copy()
        z_labels = month_labels
        label_list = month_labels
        z_title = "Month"
    else:
        filtered = gdf_local[gdf_local["week"] == selected_week].copy()
        z_labels = week_labels
        label_list = week_labels
        z_title = "Week"

    if country_mode != "both":
        filtered = filtered[filtered["country"] == country_mode]

    filtered["x_center"] = np.floor(filtered["x"]/cell_size)*cell_size + cell_size/2
    filtered["y_center"] = np.floor(filtered["y"]/cell_size)*cell_size + cell_size/2

    time_lookup = {label:i for i,label in enumerate(label_list)}
    filtered["time_index"] = filtered[time_agg].map(time_lookup)

    grouped = (
        filtered
        .groupby(["x_center","y_center","time_index",time_agg,"country"])
        .size()
        .reset_index(name="count")
    )

    geo_points = gpd.GeoDataFrame(
        grouped,
        geometry=gpd.points_from_xy(grouped["x_center"],grouped["y_center"]),
        crs=CRS_PROJECTED
    ).to_crs("EPSG:4326")

    grouped["lat"] = geo_points.geometry.y
    grouped["lon"] = geo_points.geometry.x

    max_intensity = grouped["count"].max() if not grouped.empty else 1

    x_ticks = np.linspace(grouped["x_center"].min(), grouped["x_center"].max(), 6)
    y_ticks = np.linspace(grouped["y_center"].min(), grouped["y_center"].max(), 6)

    x_geo = gpd.GeoSeries(
        gpd.points_from_xy(x_ticks,[grouped["y_center"].mean()]*len(x_ticks)),
        crs=CRS_PROJECTED).to_crs("EPSG:4326")

    y_geo = gpd.GeoSeries(
        gpd.points_from_xy([grouped["x_center"].mean()]*len(y_ticks),y_ticks),
        crs=CRS_PROJECTED).to_crs("EPSG:4326")

    x_labels = [f"{p.x:.2f}°E" for p in x_geo]
    y_labels = [f"{p.y:.2f}°N" for p in y_geo]

    cube_fig = go.Figure()

    for country,colorscale,colorbar_x in [
        ("Russia","Reds",1.02),
        ("Ukraine","Blues",1.12)
    ]:

        subset = grouped[grouped["country"] == country]

        cube_fig.add_trace(go.Scatter3d(
            x=subset["x_center"],
            y=subset["y_center"],
            z=subset["time_index"],
            mode="markers",

            customdata=np.stack(
                (subset["lat"], subset["lon"],subset["count"],subset[time_agg]),
                axis=-1
            ),

            marker=dict(
                size=8,
                color=subset["count"],
                colorscale=colorscale,
                cmin=0,
                cmax=max_intensity,
                showscale=True,
                colorbar=dict(title=f"{country} Intensity", x=colorbar_x)
            ),

            hovertemplate=
            "Lat: %{customdata[0]:.3f}<br>"
            "Lon: %{customdata[1]:.3f}<br>"
            "Strikes: %{customdata[2]}<br>"
            "%{customdata[3]}"
            "<extra></extra>"
        ))

    cube_fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(title="Longitude",tickvals=x_ticks,ticktext=x_labels),
            yaxis=dict(title="Latitude",tickvals=y_ticks,ticktext=y_labels),
            zaxis=dict(
                title=z_title,
                tickmode="array",
                tickvals=list(range(len(z_labels))),
                ticktext=z_labels
            )
        )
    )

    map_fig = go.Figure()

    for country,colorscale in [
        ("Russia","Reds"),
        ("Ukraine","Blues")
    ]:

        subset = grouped[grouped["country"] == country]

        map_fig.add_trace(go.Scattermapbox(
            lat=subset["lat"],
            lon=subset["lon"],
            mode="markers",

            customdata=np.stack(
                (subset["count"],subset[time_agg]),
                axis=-1
            ),

            marker=dict(
                size=10,
                color=subset["count"],
                colorscale=colorscale,
                cmin=0,
                cmax=max_intensity,
                showscale=True,
                colorbar=dict(
                    title=f"{country} Intensity",
                    x=1.02 if country == "Russia" else 1.12
                )
            ),

            hovertemplate=
            "Latitude: %{lat:.3f}<br>"
            "Longitude: %{lon:.3f}<br>"
            "Strikes in Bin: %{customdata[0]}<br>"
            "Date: %{customdata[1]}"
            "<extra></extra>"
        ))

    map_fig.update_layout(
        showlegend=False,
        mapbox=dict(
            style="carto-darkmatter",
            zoom=4,
            center=dict(lat=49,lon=34)
        ),
        margin=dict(l=0,r=0,t=0,b=0)
    )

    return cube_fig,map_fig


# KDE callback

BASE_RADIUS=30

@app.callback(
    Output("kde-map","figure"),
    Input("kde-map","relayoutData"),
    Input("month-filter","value")
)
def update_kde_map(relayout,selected_month):

    zoom=4
    center=dict(lat=48,lon=33)

    if relayout and "mapbox.zoom" in relayout:
        zoom=relayout["mapbox.zoom"]

    if relayout and "mapbox.center" in relayout:
        center=relayout["mapbox.center"]

    dataset=df if selected_month=="ALL" else df[df["MonthLabel"]==selected_month]

    radius=BASE_RADIUS/(zoom**0.7)

    fig=go.Figure()

    fig.add_trace(
        go.Densitymapbox(
            lat=dataset["latitude"],
            lon=dataset["longitude"],
            radius=radius,
            colorscale="Hot",
            opacity=0.7
        )
    )

    fig.update_layout(
        mapbox=dict(style="carto-darkmatter",center=center,zoom=zoom),
        uirevision="constant",
        margin=dict(l=0,r=0,t=0,b=0)
    )

    return fig

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run_server(host="0.0.0.0",port=port)
