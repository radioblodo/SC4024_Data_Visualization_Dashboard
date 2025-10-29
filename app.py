import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Optional: Streamlit internal reload noise suppression
import logging
logging.getLogger("streamlit.runtime.state").setLevel(logging.ERROR)
logging.getLogger("streamlit.web.server").setLevel(logging.ERROR)
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

# Load the CSV file into a DataFrame 
DATA_PATH = Path(__file__).parent / "data" / "athlete_events.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()
# Page setup
# -----------------------------
st.set_page_config(page_title="Olympics Data Dashboard", layout="wide")

st.title("🏅 Olympics Data Visualization Dashboard")
st.markdown("Explore Olympic trends through geographical, gender, and performance analyses.")

# -----------------------------
# Data Preparation
# -----------------------------
hosts = pd.DataFrame([
    (1896, "Athens", "Greece"),
    (1900, "Paris", "France"),
    (1904, "St. Louis", "United States"),
    (1908, "London", "United Kingdom"),
    (1912, "Stockholm", "Sweden"),
    (1920, "Antwerp", "Belgium"),
    (1924, "Paris", "France"),
    (1928, "Amsterdam", "Netherlands"),
    (1932, "Los Angeles", "United States"),
    (1936, "Berlin", "Germany"),
    (1948, "London", "United Kingdom"),
    (1952, "Helsinki", "Finland"),
    (1956, "Melbourne", "Australia"),
    (1960, "Rome", "Italy"),
    (1964, "Tokyo", "Japan"),
    (1968, "Mexico City", "Mexico"),
    (1972, "Munich", "Germany"),
    (1976, "Montreal", "Canada"),
    (1980, "Moscow", "Russia"),
    (1984, "Los Angeles", "United States"),
    (1988, "Seoul", "South Korea"),
    (1992, "Barcelona", "Spain"),
    (1996, "Atlanta", "United States"),
    (2000, "Sydney", "Australia"),
    (2004, "Athens", "Greece"),
    (2008, "Beijing", "China"),
    (2012, "London", "United Kingdom"),
    (2016, "Rio de Janeiro", "Brazil"),
], columns=["Year", "City", "Country"])

hosts["Year"] = hosts["Year"].astype(int)
hosts["info"] = hosts["City"] + " • " + hosts["Country"] + " • " + hosts["Year"].astype(str)
years_sorted = sorted(hosts["Year"].unique())


# -----------------------------
# Tabs Layout
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Geographical Analysis",
    "Gender Analysis",
    "Countries Performance",
    "Top Olympians"
])

import numpy as np

# Optional: load noc.csv once if you have it (columns: NOC, region)
# If you already loaded it earlier, just delete this block.
try:
    noc_df = pd.read_csv(r"C:\Users\zilia\.cache\kagglehub\datasets\heesoo37\120-years-of-olympic-history-athletes-and-results\versions\2\.csv")  # Kaggle file with NOC->country names
except Exception:
    noc_df = None

# Build a NOC -> CountryName mapping (fallback to a small dict if noc.csv missing)
_DEF_NOC2NAME = {
    "USA":"United States","GBR":"United Kingdom","GER":"Germany","EUN":"Russia",
    "RUS":"Russia","URS":"Russia","CHN":"China","FRA":"France","AUS":"Australia",
    "CAN":"Canada","JPN":"Japan","KOR":"South Korea","MEX":"Mexico","BRA":"Brazil",
    "GRE":"Greece","ITA":"Italy","ESP":"Spain","NED":"Netherlands","SWE":"Sweden",
    "SUI":"Switzerland","DEN":"Denmark","HUN":"Hungary","AUT":"Austria","BUL":"Bulgaria",
    "CHI":"Chile"
}
def noc_to_country(noc: pd.Series) -> pd.Series:
    if noc_df is not None and {"NOC","region"}.issubset(noc_df.columns):
        map_from_file = dict(zip(noc_df["NOC"].astype(str), noc_df["region"].astype(str)))
        return noc.astype(str).map(map_from_file).fillna(noc.astype(str).map(_DEF_NOC2NAME)).fillna(noc)
    else:
        return noc.astype(str).map(_DEF_NOC2NAME).fillna(noc)

def summer_participating_by_year(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce").astype("Int64")
    summer = d[(d["Season"] == "Summer") & (d["Year"].between(1896, 2016))]
    summer = summer[summer["Year"] != 1906]  # drop Intercalated
    by_year = (
        summer.dropna(subset=["NOC"])
              .groupby("Year")["NOC"].nunique()
              .rename("Countries").reset_index()
              .sort_values("Year")
    )
    return by_year

def choropleth_participants_for_year(df: pd.DataFrame, year: int):
    d = df.copy()
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce").astype("Int64")
    one = d[(d["Season"] == "Summer") & (d["Year"] == year)]
    if one.empty:
        raise ValueError(f"No rows for Summer {year}")
    # unique participating NOCs -> country names for Plotly
    part = (one.dropna(subset=["NOC"])["NOC"].drop_duplicates()).to_frame()
    part["Country"] = noc_to_country(part["NOC"])
    part["Participated"] = "Participated"
    fig = px.choropleth(
        part, locations="Country", locationmode="country names",
        color="Participated",
        color_discrete_map={"Participated":"#2ca02c"},
        hover_name="NOC",
        title=f"Participating Countries — Summer Olympics {year}"
    )
    fig.update_geos(showcoastlines=True, showcountries=True, countrycolor="#c7c7c7",
                    showland=True, landcolor="#f2f2f2")
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=0))
    return fig

# -----------------------------
# TAB 1: GEOGRAPHICAL ANALYSIS
# -----------------------------
with tab1:
    st.header("Geographical Analysis")

    # ---------- Shared prep ----------
    latest_year = 2016  # or: int(df.loc[df["Season"]=="Summer","Year"].max())
    marker_size = 9

    # ---------- Row A: Host map (all Summer hosts) + KPIs ----------
    colA1, colA2 = st.columns([2.2, 1])

    with colA1:
        st.subheader("Host Countries of the Summer Olympics")
        fig_hosts = px.scatter_geo(
            hosts,
            locations="Country",
            locationmode="country names",
            hover_name="City",
            hover_data={"Country": True},
            projection="natural earth",
            animation_frame="Year",          # 👈 adds the time slider
            category_orders={"Year": sorted(hosts["Year"].unique())},
            title="Summer Olympic Host Countries (1896–2016)",
        )

        fig_hosts.update_traces(marker=dict(size=10, color="#4c6ef5"))
        fig_hosts.update_layout(
            height=700,
            margin=dict(l=10, r=10, t=60, b=10),
            sliders=[dict(currentvalue={"prefix": "Year: "})],
        )

        st.plotly_chart(fig_hosts, use_container_width=True)


    with colA2:
        st.subheader("Quick Facts")
        st.metric("Total Summer Hosts", f"{hosts['Country'].nunique():,} countries")
        st.metric("Total Host Cities", f"{hosts['City'].nunique():,}")
        st.metric("Time Span", f"{hosts['Year'].min()}–{hosts['Year'].max()}")

    st.markdown("---")

   # ---------- Row 1 : Participation trend ----------
    st.subheader("Participating Countries per Summer Games (1896–2016)")

    by_year = summer_participating_by_year(df)  # returns df[Year, Countries]

    fig_bar = px.bar(
        by_year,
        x="Year", y="Countries",
        title="Countries Participating — Summer Olympics",
        labels={"Countries": "Number of Countries"},
    )
    # WWII gap shading + Cold War boycott notes
    fig_bar.add_vrect(
        x0=1940, x1=1944, line_width=0,
        fillcolor="lightgray", opacity=0.25,
        annotation_text="No Games (WWII)",
        annotation_position="top left",
    )
    for yr, txt in [(1980, "Moscow boycott"), (1984, "LA boycott")]:
        if yr in by_year["Year"].values:
            yv = int(by_year.loc[by_year["Year"] == yr, "Countries"])
            fig_bar.add_annotation(
                x=yr, y=yv, text=txt,
                showarrow=True, arrowhead=1, ay=-30
            )
    fig_bar.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # ---------- Row 2 : Map + Year slider ----------
    st.subheader("Participation Map — Select a Year")

    # slider below chart title
    min_year = int(by_year["Year"].min())
    max_year = int(by_year["Year"].max())
    default_year = max_year
    year_for_map = st.slider(
        "Year of Summer Olympics", 
        min_year, max_year, value=default_year, step=4
    )

    try:
        fig_part_map = choropleth_participants_for_year(df, year_for_map)
        fig_part_map.update_layout(
            margin=dict(l=10, r=10, t=60, b=10),
            title_text=f"Participating Countries — Summer {year_for_map}"
        )
        st.plotly_chart(fig_part_map, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.error(f"Could not render map for {year_for_map}: {e}")



# -----------------------------
# TAB 2: GENDER ANALYSIS
# -----------------------------
# =========================
# GENDER ANALYSIS TAB
# =========================
with tab2:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Ellipse
    import plotly.graph_objects as go

    # --- Session-state defaults (so charts can read them safely) ---
    all_sports = sorted(list(df["Sport"].dropna().unique()))
    years      = sorted(pd.to_numeric(df["Year"], errors="coerce").dropna().unique().astype(int))

    default_sport = "Athletics" if "Athletics" in all_sports else (all_sports[0] if all_sports else "Unknown")
    default_year  = 2016 if 2016 in years else (years[-1] if len(years) else 1896)

    # Initialize if missing
    st.session_state.setdefault("ga_sport", default_sport)
    st.session_state.setdefault("ga_smooth_k", 3)      # must be odd in your slider (1..7)
    st.session_state.setdefault("ga_donut_year", default_year)

    # If options change (e.g., filtered df), keep values valid
    if st.session_state.ga_sport not in all_sports and all_sports:
        st.session_state.ga_sport = default_sport
    if st.session_state.ga_donut_year not in years and years:
        st.session_state.ga_donut_year = default_year

    # ---------- THEME ----------
    BLUE = "#1A73E8"
    RED  = "#EA4335"

    st.header("Gender Analysis")

    # ---------- ROW A • Participation over time ----------
    # Chart (wide) | Controls (narrow)
    a_chart, a_ctrl = st.columns([4, 1.2], vertical_alignment="top")

    with a_chart:
        st.subheader("Athlete Participation by Gender (Summer, 1896 onwards)")

        d = df.copy()
        d = d[d["Season"] == "Summer"] if "Season" in d.columns else d
        d["Year"] = pd.to_numeric(d["Year"], errors="coerce").astype("Int64")
        by_year = (
            d.dropna(subset=["Year", "Sex"])
             .groupby(["Year", "Sex"]).size().unstack(fill_value=0)
        )
        by_year = by_year.loc[by_year.index >= 1896]

        # Apply smoothing set in the side column (see below)
        smooth = by_year.rolling(st.session_state.get("ga_smooth_k", 3),
                                 center=True, min_periods=1).mean().astype(float)

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(smooth.index, smooth.get("M", 0), label="Men", lw=2, color=BLUE)
        ax1.plot(smooth.index, smooth.get("F", 0), label="Women", lw=2, color=RED)

        ax1.axvline(1900, color="gray", lw=1.2, ls="--")
        ax1.axvspan(1896, 1900, color="gray", alpha=0.12)
        y_annot = float(smooth.get("F", 0).loc[1900]) if (1900 in smooth.index and "F" in smooth.columns) else 0.0
        ax1.annotate("First women's Olympics (Paris, 1900)",
                     xy=(1900, y_annot),
                     xytext=(1910, float(smooth.to_numpy().max()) * 0.25 if smooth.size else 0),
                     arrowprops=dict(arrowstyle="->", color="gray"),
                     color="gray", fontsize=10)

        ax1.set_xlabel("Year"); ax1.set_ylabel("Number of Athletes")
        ax1.set_title("Athlete Participation by Gender")
        ax1.legend(title="Gender"); ax1.grid(alpha=0.2)
        st.pyplot(fig1, use_container_width=True)

    with a_ctrl:
        st.markdown("**Display**")
        smooth_k = st.slider("Smoothing (years)", 1, 7, 3, step=2,
                             key="ga_smooth_k",
                             help="Centered rolling window.")
        # Quick stats block
        if not by_year.empty:
            total_m = int(by_year.get("M", 0).sum())
            total_f = int(by_year.get("F", 0).sum())
            st.markdown(
                f"""
                <div style="background:#F6F8FB;border:1px solid #E5EAF2;border-radius:8px;padding:10px">
                <b>Totals</b><br>
                Men: {total_m:,}<br>
                Women: {total_f:,}
                </div>
                """, unsafe_allow_html=True
            )

    st.markdown("---")

    # ---------- ROW B • Donut (left) ----------
    b_chart_left, b_ctrl_left = st.columns([3, 1.2], vertical_alignment="top")

    with b_chart_left:
        st.subheader("Gender Composition — Selected Summer Games")
        donut_year = st.session_state.get("ga_donut_year", 2016)
        d2 = d[d["Year"] == donut_year].copy()

        if d2.empty:
            st.info(f"No Summer rows found for {donut_year}.")
        else:
            gender_counts = (d2["Sex"].value_counts()
                               .reindex(["M", "F"]).fillna(0).reset_index())
            gender_counts.columns = ["Sex", "Count"]

            fig_donut = go.Figure(
                data=[go.Pie(
                    labels=gender_counts["Sex"].map({"M": "Male", "F": "Female"}),
                    values=gender_counts["Count"],
                    hole=0.55, textinfo="label+percent",
                    marker=dict(colors=[BLUE, RED])
                )]
            )
            fig_donut.add_annotation(text=f"<b>{donut_year}<br>Gender Mix</b>",
                                     x=0.5, y=0.5, showarrow=False, font_size=16)
            fig_donut.update_layout(showlegend=False,
                                    margin=dict(t=10, b=10, l=10, r=10),
                                    height=420)
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    with b_ctrl_left:
        st.markdown("**Select Year**")
        yrs = sorted(d["Year"].dropna().unique())
        default_idx = yrs.index(2016) if 2016 in yrs else len(yrs) - 1
        donut_year = st.selectbox("Summer Year", yrs, index=default_idx, key="ga_donut_year")

    # ---------- ROW B • Scatter (right) ----------
    b_chart_right, b_ctrl_right = st.columns([3, 1.2], vertical_alignment="top")

    with b_chart_right:
        # sport_name is chosen in the right control column
        st.subheader(f"{st.session_state.get('ga_sport', 'Athletics')}: Medalists — Height vs Weight")

        valid_medals = {"Gold", "Silver", "Bronze"}
        subset = (
            df[df["Medal"].isin(valid_medals)]
            .dropna(subset=["Height", "Weight", "Sex", "Sport"])
            .query("Sport == @st.session_state.ga_sport")
        )

        if subset.empty:
            st.info(f"No medalist height/weight data for {st.session_state.get('ga_sport', 'this sport')}.")
        else:
            palette = {'M': BLUE, 'F': RED}
            # ↓ wide and shallow
            fig2, ax2 = plt.subplots(figsize=(6, 5))   # (width_in, height_in)
            # tidy margins so there’s no wasted padding
            fig2.subplots_adjust(top=0.88, bottom=0.20, left=0.12, right=0.98)
            sns.scatterplot(data=subset, x="Height", y="Weight", hue="Sex",
                            s=45, ax=ax2, palette=palette, alpha=0.85, linewidth=0)

            # Ellipse enclosure per gender
            for sex, g in subset.groupby("Sex"):
                x, y = g["Height"].to_numpy(float), g["Weight"].to_numpy(float)
                if len(x) >= 3:
                    mx, my, sx, sy = x.mean(), y.mean(), x.std(ddof=0), y.std(ddof=0)
                    ax2.add_patch(Ellipse((mx, my), 2*sx*1.6, 2*sy*1.6,
                                          fill=False, lw=2, edgecolor=palette.get(sex, "gray"), alpha=0.9))

            ax2.set_xlabel("Height (cm)"); ax2.set_ylabel("Weight (kg)")
            ax2.set_title(f"{st.session_state.ga_sport}: Medalists Height vs Weight")
            ax2.grid(alpha=0.2); ax2.legend(title="Sex", loc="best")
            st.pyplot(fig2, use_container_width=True)

    with b_ctrl_right:
        st.markdown("**Choose Sport**")
        all_sports = sorted(list(df["Sport"].dropna().unique()))
        default_idx = all_sports.index("Athletics") if "Athletics" in all_sports else 0
        st.selectbox("Sport", all_sports, index=default_idx, key="ga_sport")


# -----------------------------
# TAB 3: COUNTRIES PERFORMANCE
# -----------------------------
# =========================
# Inside your Countries Performance tab
# =========================
with tab3:
    st.header("Countries Performance and Medal Distribution")

    # --- Prep data (all medals, both seasons) ---
    medals = df[df["Medal"].isin(["Gold", "Silver", "Bronze"])].copy()
    medals["Year"] = pd.to_numeric(medals["Year"], errors="coerce").astype("Int64")

    # --- KPI row (no selectors needed) ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Medals", f"{39772}")
    c2.metric("Countries with Medals", f"{medals['NOC'].nunique():,}")
    # unique athletes might be named 'ID' in athlete_events.csv
    unique_ath = medals["ID"].nunique() if "ID" in medals.columns else pd.NA
    c3.metric("Unique Medalists", f"{28202}" if pd.notna(unique_ath) else "—")
    year_min, year_max = int(medals["Year"].min()), int(medals["Year"].max())
    c4.metric("Coverage (Years)", f"{year_min}–{year_max}")

    st.subheader("Overview of Medal Distribution by Country")

    # --- Row 1: histogram (left) + top countries barh (right) ---
    left, right = st.columns([1, 1])
    with left:
        # medals per country distribution (how concentrated the medals are)
        by_country = medals.groupby("NOC")["Medal"].count().rename("Medals").reset_index()
        fig_hist = px.histogram(
            by_country, x="Medals", nbins=40,
            title="Distribution of Medals per Country"
        )
        fig_hist.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)
    with right:
        # top 10 countries by total medals
        top10 = by_country.sort_values("Medals", ascending=False).head(10).sort_values("Medals")
        # breakdown for top 10 countries (same set as above)
        top_set = set(top10["NOC"])
        stack = (
            medals[medals["NOC"].isin(top_set)]
            .groupby(["NOC", "Medal"]).size().reset_index(name="Count")
        )
        # keep same country order as top10
        stack["NOC"] = pd.Categorical(stack["NOC"], categories=list(top10["NOC"]), ordered=True)
        fig_stack = px.bar(
            stack.sort_values(["NOC", "Medal"]),
            x="NOC", y="Count", color="Medal",
            title="Medal Breakdown for Top 10 Countries",
            barmode="stack"
        )
        fig_stack.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_stack, use_container_width=True)

    # --- Row 2: Medal Composition (%) + Cumulative Medals Trend ---
    st.subheader("Medal Composition and Trends Over Time")
    left2, right2 = st.columns(2)
    with left2:
        comp = (
            medals[medals["NOC"].isin(top10["NOC"])]
            .groupby(["NOC", "Medal"]).size().unstack(fill_value=0).reset_index()
        )

        fig_balance = px.scatter(
            comp,
            x="Gold", y="Silver",
            size="Bronze", text="NOC",
            color="Gold",
            color_continuous_scale="YlGnBu",
            title="Medal Composition Balance (Top 10 Countries)",
        )
        fig_balance.update_traces(textposition="top center")
        fig_balance.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_balance, use_container_width=True)
    with right2:
        # --- Cumulative Medals Over Time ---
        default_countries = list(top10.sort_values("Medals", ascending=False)["NOC"].head(5))


        by_year_country = (
            medals[medals["NOC"].isin(default_countries)]
            .dropna(subset=["Year"])
            .groupby(["NOC", "Year"]).size().reset_index(name="Medals")
            .sort_values(["NOC", "Year"])
        )

        # Cumulative medals
        by_year_country["Cumulative"] = by_year_country.groupby("NOC")["Medals"].cumsum()

        fig_cum = px.line(
            by_year_country,
            x="Year", y="Cumulative", color="NOC",
            markers=True,
            title="Cumulative Medals Over Time (Top 5 Countries)"
        )
        fig_cum.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_cum, use_container_width=True)

    # Row 3
    # --- Region: Southeast Asia ---
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    from matplotlib.patches import Rectangle
    import streamlit as st

    # --- Southeast Asia countries to include ---
    sea_countries = [
        'Singapore', 'Malaysia', 'Thailand', 'Indonesia', 'Philippines',
        'Vietnam', 'Myanmar', 'Cambodia', 'Laos', 'Brunei', 'Timor-Leste'
    ]

    st.subheader("Southeast Asia Countries Performance")

    # --- Preprocess ---
    df = df.drop_duplicates()
    df_medals = df[df['Medal'].notna()].copy()
    df_sea = df_medals[df_medals['Team'].isin(sea_countries)].copy()

    unique_medals = df_sea.drop_duplicates(subset=['Team', 'Games', 'Event', 'Medal'])

    medal_counts = (
        unique_medals['Team']
        .value_counts()
        .reindex(sea_countries, fill_value=0)
        .sort_values(ascending=True)
    )

    # --- Gradient color mapping ---
    cmap = cm.get_cmap('BuPu')  # Yellow-Green-Blue gradient
    norm = mcolors.Normalize(vmin=medal_counts.min(), vmax=medal_counts.max())
    colors = [cmap(norm(v)) for v in medal_counts.values]

    # --- Plot ---
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        medal_counts.index,
        medal_counts.values,
        color=colors,
        edgecolor='black',
        linewidth=0.8,
        zorder=3
    )

    ax.yaxis.grid(True, linestyle=':', linewidth=0.8, color='gray', zorder=0)
    ax.spines['left'].set_linestyle(':')
    ax.spines['left'].set_linewidth(1)
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('Country', fontsize=12)
    ax.set_ylabel('Medal Count', fontsize=12)
    ax.set_title('Total Olympic Medals by Southeast Asian Countries', fontsize=14)
    plt.xticks(rotation=45, ha='right')

    # Annotate bar values
    max_val = medal_counts.values.max()
    y_offset = max(0.1, max_val * 0.02)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + y_offset,
            f'{int(h)}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    ax.margins(y=0.10)

    # --- Highlight Top 3 ---
    if len(bars) >= 3:
        left_bar = bars[-3]
        right_bar = bars[-1]

        x0 = left_bar.get_x()
        x1 = right_bar.get_x() + right_bar.get_width()
        pad = left_bar.get_width() * 0.35
        x0 -= pad
        x1 += pad

        ymin, ymax = ax.get_ylim()
        y0 = ymin - (ymax - ymin) * 0.03
        y1 = ymax + (ymax - ymin) * 0.03

        rect = Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor='lightblue',
            alpha=0.20,
            edgecolor='royalblue',
            linewidth=2,
            zorder=1,
            clip_on=False
        )
        ax.add_patch(rect)
        ax.text(
            (x0 + x1) / 2, y1,
            'Top 3',
            ha='center', va='bottom',
            fontsize=11,
            color='royalblue'
        )

    # --- Add colorbar legend ---
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Medal Count', rotation=270, labelpad=15)

    plt.tight_layout()
    st.pyplot(fig)




# -----------------------------
# TAB 4: TOP OLYMPIANS
# -----------------------------
# =========================
# TAB 4: TOP OLYMPIANS
# =========================
with tab4:
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    st.header("Top Olympians Analysis")

    # -----------------------------
    # Data prep (robust to missing cols)
    # -----------------------------
    if not {"Name","Medal","Sport","Year","NOC"}.issubset(df.columns):
        st.error("Top Olympians requires columns: Name, Medal, Sport, Year, NOC.")
        st.stop()

    medal_colors = {"Gold":"#FFD700", "Silver":"#C0C0C0", "Bronze":"#CD7F32"}
    df_ = df.copy()
    df_["Year"] = pd.to_numeric(df_["Year"], errors="coerce").astype("Int64")
    df_medals = df_.dropna(subset=["Medal"])  # keep only rows with a medal
    df_medals = df_medals[df_medals["Medal"].isin(["Gold","Silver","Bronze"])]

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    total_medals = len(df_medals)
    unique_medalists = df_medals["Name"].nunique()
    most_decorated = (
        df_medals.groupby("Name")["Medal"].count()
        .sort_values(ascending=False).head(1)
    )
    top_name = most_decorated.index[0] if len(most_decorated) else "—"
    top_count = int(most_decorated.iloc[0]) if len(most_decorated) else 0
    sports_with_medals = df_medals["Sport"].nunique()

    k1.metric("Total Medals", f"{total_medals:,}")
    k2.metric("Unique Medalists", f"{unique_medalists:,}")
    k3.metric("Most Decorated", f"{top_name} ({top_count})")
    k4.metric("Sports with Medals", f"{sports_with_medals:,}")

    st.markdown("---")

    # -----------------------------
    # CHART 1: Top N Olympians by TOTAL medals (stacked by type)
    # -----------------------------
    st.subheader("Top Olympians by Total Medals")
    topN = 10  # you can turn this into a slider if you like

    # medals by (Name, Medal)
    stacked = (
        df_medals.groupby(["Name", "Medal"]).size()
        .reset_index(name="Count")
    )

    # total medals per athlete
    total = stacked.groupby("Name")["Count"].sum().sort_values(ascending=False).head(topN)
    top_names = total.index.tolist()

    # keep only topN and sort descending
    stacked_top = stacked[stacked["Name"].isin(top_names)]

    # categorical order: descending (top -> bottom)
    stacked_top["Name"] = pd.Categorical(
        stacked_top["Name"],
        categories=top_names,  # already descending
        ordered=True
    )

    # plot — ensure descending order by sorting values before plotting
    fig_total = px.bar(
        stacked_top.sort_values(["Name", "Medal"], ascending=[False, True]),
        x="Count", y="Name", color="Medal",
        color_discrete_map=medal_colors,
        title=f"Top {topN} Olympians by Total Medals",
        orientation="h", barmode="stack"
    )

    # Make sure axis respects the category order
    fig_total.update_layout(
        yaxis=dict(categoryorder="array", categoryarray=top_names[::-1]),
        margin=dict(l=10, r=10, t=60, b=10),
        height=550
    )

    st.plotly_chart(fig_total, use_container_width=True)


    # -----------------------------
    # CHART 2: Top N by GOLD medals
    # -----------------------------
    c21, c22 = st.columns([1,1])

    with c21:
        st.subheader("Age Distribution of Top Olympians")

        # --- Find top Olympians by total medals ---
        topN = 10
        total_medals = (
            df_medals.groupby("Name")["Medal"]
            .count()
            .sort_values(ascending=False)
            .head(topN)
            .index
        )

        # --- Filter dataset to these athletes ---
        top_athletes = df_medals[df_medals["Name"].isin(total_medals)].copy()
        top_athletes = top_athletes.dropna(subset=["Age"])

        # --- Create box plot of Age per athlete ---
        fig_age = px.box(
            top_athletes,
            x="Name",
            y="Age",
            color="Name",
            title=f"Age Distribution of Top {topN} Olympians",
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        fig_age.update_layout(
            xaxis_title="Olympian",
            yaxis_title="Age",
            showlegend=False,
            margin=dict(l=10, r=10, t=60, b=10),
            height=450
        )

        st.plotly_chart(fig_age, use_container_width=True)


    # -----------------------------
    # CHART 3: For a selected sport — top medalists (stacked)
    # -----------------------------
    with c22:
        st.subheader("Top Medalists — Physical Attributes by Sport")

        # --- Sport selector ---
        sports_sorted = sorted(df_medals["Sport"].dropna().unique())
        sport_sel = st.selectbox(
            "Choose a sport", 
            sports_sorted,
            index=sports_sorted.index("Swimming") if "Swimming" in sports_sorted else 0,
            key="sport_violin_sel"
        )

        # --- Get top 10 medalists in the selected sport ---
        sport_medals = df_medals[df_medals["Sport"] == sport_sel].copy()
        sport_stack = (
            sport_medals.groupby(["Name", "Medal"])
            .size()
            .reset_index(name="Count")
        )
        sport_total = (
            sport_stack.groupby("Name")["Count"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        top_names = sport_total.index.tolist()

        # --- Filter top athletes with height & weight data ---
        top_athletes = sport_medals[
            sport_medals["Name"].isin(top_names)
        ].dropna(subset=["Height", "Weight"])

        if top_athletes.empty:
            st.info(f"No height/weight data available for top medalists in {sport_sel}.")
        else:
            sex_colors = {"M": "#1A73E8", "F": "#EA4335"}

            # --- Side-by-side layout for Height & Weight violins ---
            col_h, col_w = st.columns(2)

            # Height distribution
            with col_h:
                fig_h = px.violin(
                    top_athletes,
                    x="Sex", y="Height", color="Sex",
                    points="all",  # keep individual points, remove boxplot
                    color_discrete_map=sex_colors,
                    title=f"{sport_sel}: Height Distribution",
                )
                fig_h.update_layout(
                    xaxis_title="Sex",
                    yaxis_title="Height (cm)",
                    margin=dict(l=10, r=10, t=60, b=10),
                    legend_title="Sex",
                    height=400,
                )
                st.plotly_chart(fig_h, use_container_width=True)

            # Weight distribution
            with col_w:
                fig_w = px.violin(
                    top_athletes,
                    x="Sex", y="Weight", color="Sex",
                    points="all",  # keep points, remove boxplot
                    color_discrete_map=sex_colors,
                    title=f"{sport_sel}: Weight Distribution",
                )
                fig_w.update_layout(
                    xaxis_title="Sex",
                    yaxis_title="Weight (kg)",
                    margin=dict(l=10, r=10, t=60, b=10),
                    legend_title="Sex",
                    height=400,
                )
                st.plotly_chart(fig_w, use_container_width=True)






